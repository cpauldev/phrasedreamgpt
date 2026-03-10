import * as ort from "onnxruntime-web";

const CONFIG = Object.freeze({
  bundleMagic: "PDBGONNX",
  bundleFormat: "dreamphrasegpt-onnx-bundle",
  bundleVersion: 1,
  maxRetries: 40,
  sourceFilterKind: "bloom",
  sourceFilterVersion: 1,
});

type Tokenizer = {
  blockSize: number;
  bosId: number;
  idToChar: string[];
  vocabSize: number;
};

type SourceFilter = {
  bitCount: number;
  bits: Uint8Array;
  hashCount: number;
};

type NumericTypedArray =
  | Float32Array
  | Float64Array
  | Int8Array
  | Int16Array
  | Int32Array
  | Uint8Array
  | Uint8ClampedArray
  | Uint16Array
  | Uint32Array;

type ModelRuntime = {
  session: ort.InferenceSession;
  sourceFilter: SourceFilter | null;
  tokenizer: Tokenizer;
};

type GenerateOptions = {
  signal?: AbortSignal;
  samples: number;
  temperature: number;
};

const runtimeCache = new Map<string, Promise<ModelRuntime>>();
const decoder = new TextDecoder();
const encoder = new TextEncoder();

export async function generateBundleSamples(relativeBundlePath: string, options: GenerateOptions) {
  validateGenerateOptions(options);
  throwIfAborted(options.signal);
  configureOrt();
  const runtime = await getRuntime(relativeBundlePath);
  throwIfAborted(options.signal);
  return generateFilteredSamples(runtime, options);
}

async function getRuntime(relativeBundlePath: string) {
  const bundleUrl = resolveAssetUrl(relativeBundlePath);
  const cached = runtimeCache.get(bundleUrl);
  if (cached) {
    return cached;
  }

  const runtimePromise = loadRuntime(bundleUrl);
  runtimeCache.set(bundleUrl, runtimePromise);

  try {
    return await runtimePromise;
  } catch (error) {
    runtimeCache.delete(bundleUrl);
    throw error;
  }
}

function resolveAssetUrl(relativePath: string) {
  const baseUrl = new URL(import.meta.env.BASE_URL, window.location.href);
  return new URL(relativePath, baseUrl).toString();
}

let ortConfigured = false;

function configureOrt() {
  if (ortConfigured) {
    return;
  }

  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ort.env.wasm.wasmPaths = {
    mjs: resolveAssetUrl("ort/ort-wasm-simd-threaded.mjs"),
    wasm: resolveAssetUrl("ort/ort-wasm-simd-threaded.wasm"),
  };
  ortConfigured = true;
}

async function loadRuntime(bundleUrl: string): Promise<ModelRuntime> {
  const bundleBytes = await fetchBundle(bundleUrl);
  const { modelBytes, sourceFilter, tokenizer } = parseBundle(bundleBytes, bundleUrl);
  const session = await ort.InferenceSession.create(modelBytes, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });

  return { session, sourceFilter, tokenizer };
}

async function fetchBundle(bundleUrl: string) {
  let response: Response;

  try {
    response = await fetch(bundleUrl);
  } catch {
    throw new Error("Failed to load model bundle. Check your connection and try again.");
  }

  if (!response.ok) {
    throw new Error(`Failed to load model bundle: ${response.status} ${response.statusText}`);
  }

  return new Uint8Array(await response.arrayBuffer());
}

function parseBundle(bundleBytes: Uint8Array, bundleUrl: string) {
  if (bundleBytes.length < CONFIG.bundleMagic.length + 4) {
    throw new Error(`Corrupt DreamPhraseGPT bundle: ${bundleUrl}`);
  }

  const magic = decoder.decode(bundleBytes.subarray(0, CONFIG.bundleMagic.length));
  if (magic !== CONFIG.bundleMagic) {
    throw new Error(`Invalid DreamPhraseGPT bundle: ${bundleUrl}`);
  }

  const view = new DataView(bundleBytes.buffer, bundleBytes.byteOffset, bundleBytes.byteLength);
  const headerLength = view.getUint32(CONFIG.bundleMagic.length, true);
  const headerStart = CONFIG.bundleMagic.length + 4;
  const headerEnd = headerStart + headerLength;

  if (headerEnd > bundleBytes.length) {
    throw new Error(`Corrupt DreamPhraseGPT bundle header: ${bundleUrl}`);
  }

  let header: Record<string, unknown>;
  try {
    header = JSON.parse(decoder.decode(bundleBytes.subarray(headerStart, headerEnd)));
  } catch {
    throw new Error(`Corrupt DreamPhraseGPT bundle header JSON: ${bundleUrl}`);
  }

  if (header.format !== CONFIG.bundleFormat) {
    throw new Error(`Unsupported bundle format: ${String(header.format ?? "<missing>")}`);
  }
  if (header.version !== CONFIG.bundleVersion) {
    throw new Error(`Unsupported bundle version: ${String(header.version ?? "<missing>")}`);
  }

  const modelBytes = bundleBytes.slice(headerEnd);
  if (modelBytes.length === 0) {
    throw new Error(`Bundle is missing ONNX model bytes: ${bundleUrl}`);
  }

  return {
    modelBytes,
    sourceFilter: validateSourceFilter(header.source_filter, bundleUrl),
    tokenizer: validateTokenizer(header.tokenizer, bundleUrl),
  };
}

function validateTokenizer(tokenizerValue: unknown, bundleUrl: string): Tokenizer {
  if (!tokenizerValue || typeof tokenizerValue !== "object") {
    throw new Error(`Bundle is missing tokenizer metadata: ${bundleUrl}`);
  }

  const tokenizer = tokenizerValue as Record<string, unknown>;
  const idToChar = tokenizer.id_to_char;
  const bosId = tokenizer.bos_id;
  const blockSize = tokenizer.block_size;
  const vocabSize = tokenizer.vocab_size;
  const validBosId = readInteger(bosId, `Bundle tokenizer metadata is incomplete: ${bundleUrl}`);
  const validBlockSize = readPositiveInteger(
    blockSize,
    `Bundle tokenizer metadata is incomplete: ${bundleUrl}`,
  );
  const validVocabSize = readPositiveInteger(
    vocabSize,
    `Bundle tokenizer metadata is incomplete: ${bundleUrl}`,
  );

  if (!Array.isArray(idToChar) || !idToChar.every((item) => typeof item === "string")) {
    throw new Error(`Bundle tokenizer has an invalid id_to_char table: ${bundleUrl}`);
  }
  if (validBosId !== idToChar.length || validVocabSize !== idToChar.length + 1) {
    throw new Error(`Bundle tokenizer metadata is inconsistent: ${bundleUrl}`);
  }

  return {
    blockSize: validBlockSize,
    bosId: validBosId,
    idToChar,
    vocabSize: validVocabSize,
  };
}

function validateSourceFilter(sourceFilterValue: unknown, bundleUrl: string): SourceFilter | null {
  if (sourceFilterValue == null) {
    return null;
  }
  if (typeof sourceFilterValue !== "object") {
    throw new Error(`Bundle source filter is invalid: ${bundleUrl}`);
  }

  const sourceFilter = sourceFilterValue as Record<string, unknown>;
  const kind = sourceFilter.kind;
  const version = sourceFilter.version;
  const bitCount = sourceFilter.bit_count;
  const hashCount = sourceFilter.hash_count;
  const bitsBase64 = sourceFilter.bits_base64;
  const validBitCount = readPositiveInteger(
    bitCount,
    `Bundle source filter has an invalid bit count: ${bundleUrl}`,
  );
  const validHashCount = readPositiveInteger(
    hashCount,
    `Bundle source filter has an invalid hash count: ${bundleUrl}`,
  );

  if (kind !== CONFIG.sourceFilterKind || version !== CONFIG.sourceFilterVersion) {
    throw new Error(`Unsupported source filter payload in ${bundleUrl}`);
  }
  if (typeof bitsBase64 !== "string" || bitsBase64.length === 0) {
    throw new Error(`Bundle source filter is missing bit payload: ${bundleUrl}`);
  }

  const bits = decodeBase64(bitsBase64);
  const expectedBytes = Math.ceil(validBitCount / 8);
  if (bits.length !== expectedBytes) {
    throw new Error(`Bundle source filter has the wrong bit payload size: ${bundleUrl}`);
  }

  return {
    bitCount: validBitCount,
    bits,
    hashCount: validHashCount,
  };
}

function decodeBase64(value: string) {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);

  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }

  return bytes;
}

function readInteger(value: unknown, errorMessage: string): number {
  if (!Number.isInteger(value)) {
    throw new Error(errorMessage);
  }

  return Number(value);
}

function readPositiveInteger(value: unknown, errorMessage: string): number {
  const parsed = readInteger(value, errorMessage);

  if (parsed <= 0) {
    throw new Error(errorMessage);
  }

  return parsed;
}

function sampleLogitIndex(logits: number[], temperature: number) {
  const scaledTemperature = 1 / temperature;
  let maxScaledLogit = Number.NEGATIVE_INFINITY;

  for (let index = 0; index < logits.length; index += 1) {
    const logit = logits[index];
    if (logit === undefined) {
      throw new Error("Logits array is missing a value during sampling.");
    }

    const scaledLogit = logit * scaledTemperature;
    if (scaledLogit > maxScaledLogit) {
      maxScaledLogit = scaledLogit;
    }
  }

  const weights = new Array<number>(logits.length);
  let totalWeight = 0;

  for (let index = 0; index < logits.length; index += 1) {
    const logit = logits[index];
    if (logit === undefined) {
      throw new Error("Logits array is missing a value during sampling.");
    }

    const weight = Math.exp(logit * scaledTemperature - maxScaledLogit);
    weights[index] = weight;
    totalWeight += weight;
  }

  let remaining = Math.random() * totalWeight;

  for (let index = 0; index < weights.length; index += 1) {
    const weight = weights[index];
    if (weight === undefined) {
      throw new Error("Weight array is missing a value during sampling.");
    }

    remaining -= weight;
    if (remaining <= 0) {
      return index;
    }
  }

  return weights.length - 1;
}

function getLastLogits(
  logitsTensor: ort.InferenceSession.ReturnType[string],
  sequenceLength: number,
  expectedVocabSize: number,
): number[] {
  if (!logitsTensor || !Array.isArray(logitsTensor.dims)) {
    throw new Error('ONNX session returned an invalid "logits" tensor.');
  }

  const [batchSize, outputSequenceLength, vocabSize] = logitsTensor.dims;
  if (logitsTensor.dims.length !== 3) {
    throw new Error(`Unexpected ONNX logits rank ${logitsTensor.dims.length}.`);
  }
  if (
    batchSize !== 1 ||
    outputSequenceLength !== sequenceLength ||
    vocabSize !== expectedVocabSize
  ) {
    throw new Error("ONNX session returned logits with unexpected dimensions.");
  }

  const data = logitsTensor.data;
  if (!data) {
    throw new Error('ONNX session returned a "logits" tensor without readable data.');
  }

  const offset = (sequenceLength - 1) * vocabSize;
  return sliceNumericTensorData(data, offset, vocabSize);
}

async function generateOneSample(runtime: ModelRuntime, temperature: number, signal?: AbortSignal) {
  const { tokenizer } = runtime;
  const tokenIds = [tokenizer.bosId];
  const characters: string[] = [];

  for (let step = 0; step < tokenizer.blockSize; step += 1) {
    throwIfAborted(signal);
    const window = tokenIds.slice(-tokenizer.blockSize);
    const input = new ort.Tensor(
      "int64",
      BigInt64Array.from(window.map((tokenId) => BigInt(tokenId))),
      [1, window.length],
    );
    const output = await runtime.session.run({ idx: input });
    throwIfAborted(signal);
    const logits = output.logits;

    if (!logits) {
      throw new Error('ONNX session output is missing "logits".');
    }

    const tokenId = sampleLogitIndex(
      getLastLogits(logits, window.length, tokenizer.vocabSize),
      temperature,
    );

    if (tokenId === tokenizer.bosId) {
      break;
    }

    const nextCharacter = tokenizer.idToChar[tokenId];
    if (typeof nextCharacter !== "string") {
      throw new Error("ONNX session returned a token that is outside the tokenizer range.");
    }

    characters.push(nextCharacter);
    tokenIds.push(tokenId);
  }

  return characters.join("");
}

async function generateFilteredSamples(runtime: ModelRuntime, options: GenerateOptions) {
  const results: string[] = [];

  for (let sampleIndex = 0; sampleIndex < options.samples; sampleIndex += 1) {
    throwIfAborted(options.signal);
    let accepted: string | null = null;

    for (let attempt = 0; attempt < CONFIG.maxRetries; attempt += 1) {
      throwIfAborted(options.signal);
      const candidate = await generateOneSample(runtime, options.temperature, options.signal);
      if (candidate.trim().length === 0) {
        continue;
      }
      if (!(await sourceFilterMatches(runtime.sourceFilter, candidate, options.signal))) {
        accepted = candidate;
        break;
      }
    }

    if (!accepted) {
      throw new Error(`Failed to sample a non-source line within ${CONFIG.maxRetries} attempts.`);
    }

    results.push(accepted);
  }

  return results;
}

async function sourceFilterMatches(
  sourceFilter: SourceFilter | null,
  text: string,
  signal?: AbortSignal,
) {
  if (!sourceFilter) {
    return false;
  }

  const normalized = text.trim();
  if (normalized.length === 0) {
    return false;
  }

  throwIfAborted(signal);
  const digest = await crypto.subtle.digest("SHA-256", encoder.encode(normalized));
  throwIfAborted(signal);
  const digestView = new DataView(digest);
  const bitCount = BigInt(sourceFilter.bitCount);
  const first = digestView.getBigUint64(0, true);
  const secondRaw = digestView.getBigUint64(8, true);
  const second = secondRaw === 0n ? 0x9e3779b97f4a7c15n : secondRaw;

  for (let index = 0n; index < BigInt(sourceFilter.hashCount); index += 1n) {
    const bitIndex = Number((first + index * second) % bitCount);
    const byteIndex = Math.floor(bitIndex / 8);
    const bitOffset = bitIndex % 8;

    if ((sourceFilter.bits[byteIndex] & (1 << bitOffset)) === 0) {
      return false;
    }
  }

  return true;
}

function sliceNumericTensorData(
  data: ort.Tensor.DataTypeMap[ort.Tensor.Type] | ort.InferenceSession.ReturnType[string]["data"],
  offset: number,
  length: number,
): number[] {
  if (Array.isArray(data)) {
    if (!data.every((value) => typeof value === "number")) {
      throw new Error('ONNX session returned a "logits" tensor with non-numeric data.');
    }

    return data.slice(offset, offset + length);
  }

  if (!isNumericTypedArray(data)) {
    throw new Error('ONNX session returned a "logits" tensor with non-numeric data.');
  }

  return Array.from(data.subarray(offset, offset + length), Number);
}

function isNumericTypedArray(value: unknown): value is NumericTypedArray {
  return (
    value instanceof Float32Array ||
    value instanceof Float64Array ||
    value instanceof Int8Array ||
    value instanceof Int16Array ||
    value instanceof Int32Array ||
    value instanceof Uint8Array ||
    value instanceof Uint8ClampedArray ||
    value instanceof Uint16Array ||
    value instanceof Uint32Array
  );
}

function validateGenerateOptions(options: GenerateOptions) {
  if (!Number.isInteger(options.samples) || options.samples <= 0) {
    throw new Error("Sample count must be a positive integer.");
  }

  if (!Number.isFinite(options.temperature) || options.temperature <= 0) {
    throw new Error("Temperature must be a positive number.");
  }
}

function throwIfAborted(signal?: AbortSignal) {
  if (!signal?.aborted) {
    return;
  }

  throw createAbortError();
}

function createAbortError() {
  if (typeof DOMException !== "undefined") {
    return new DOMException("Generation was cancelled.", "AbortError");
  }

  const error = new Error("Generation was cancelled.");
  error.name = "AbortError";
  return error;
}
