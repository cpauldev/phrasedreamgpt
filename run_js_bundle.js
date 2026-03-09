/**
 * run_js_bundle.js — run a DreamPhraseGPT JS bundle exported by `dreamphrasegpt.py`
 *
 * Usage:
 *   node run_js_bundle.js
 *   node run_js_bundle.js us_baby_names.model
 *   node run_js_bundle.js us_baby_names.model --samples 30 --temperature 0.7
 *
 * If no bundle path is provided, the newest `*.model` file anywhere in `models/` is used.
 */

const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const ort = require("onnxruntime-node");

const CONFIG = Object.freeze({
  modelsDir: path.resolve(__dirname, "models"),
  defaultSamples: 20,
  defaultTemperature: 0.8,
  sourceFilterMaxRetries: 40,
  bundleMagic: "PDBGONNX",
  bundleFormat: "dreamphrasegpt-onnx-bundle",
  bundleVersion: 1,
  sourceFilterKind: "bloom",
  sourceFilterVersion: 1,
});

const USAGE_TEXT = [
  "Usage:",
  "  node run_js_bundle.js [bundle-path] [--samples N] [--temperature FLOAT]",
  "",
  "Examples:",
  "  node run_js_bundle.js",
  "  node run_js_bundle.js us_baby_names.model",
  "  node run_js_bundle.js us_baby_names.model --samples 40 --temperature 0.7",
].join("\n");

function printUsage() {
  console.log(USAGE_TEXT);
}

function readFlagValue(argv, index, flagName) {
  const arg = argv[index];
  const inlinePrefix = `${flagName}=`;

  if (arg.startsWith(inlinePrefix)) {
    return { value: arg.slice(inlinePrefix.length), nextIndex: index };
  }

  if (arg === flagName) {
    if (index + 1 >= argv.length) {
      throw new Error(`${flagName} requires a value.`);
    }
    return { value: argv[index + 1], nextIndex: index + 1 };
  }

  return null;
}

function parseArgs(argv) {
  const options = {
    help: false,
    bundlePath: null,
    samples: CONFIG.defaultSamples,
    temperature: CONFIG.defaultTemperature,
  };

  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index];

    if (arg === "--help" || arg === "-h") {
      options.help = true;
      return options;
    }

    const samplesValue = readFlagValue(argv, index, "--samples");
    if (samplesValue !== null) {
      options.samples = Number.parseInt(samplesValue.value, 10);
      index = samplesValue.nextIndex;
      continue;
    }

    const temperatureValue = readFlagValue(argv, index, "--temperature");
    if (temperatureValue !== null) {
      options.temperature = Number.parseFloat(temperatureValue.value);
      index = temperatureValue.nextIndex;
      continue;
    }

    if (arg.startsWith("--")) {
      throw new Error(`Unknown option: ${arg}`);
    }

    if (options.bundlePath !== null) {
      throw new Error(`Unexpected extra argument: ${arg}`);
    }

    options.bundlePath = arg;
  }

  if (!Number.isInteger(options.samples) || options.samples < 0) {
    throw new Error("--samples must be a non-negative integer.");
  }

  if (!Number.isFinite(options.temperature) || options.temperature <= 0) {
    throw new Error("--temperature must be greater than 0.");
  }

  return options;
}

function isBundleFile(filePath) {
  return filePath.endsWith(".model") && !filePath.endsWith(".model.pt");
}

function collectBundlePaths(directory) {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const entryPath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      return collectBundlePaths(entryPath);
    }
    if (entry.isFile() && isBundleFile(entry.name)) {
      return [entryPath];
    }
    return [];
  });
}

function listBundles(modelsDir) {
  if (!fs.existsSync(modelsDir)) {
    return [];
  }

  return collectBundlePaths(modelsDir)
    .map((bundlePath) => ({ bundlePath, mtimeMs: fs.statSync(bundlePath).mtimeMs }))
    .sort((left, right) => right.mtimeMs - left.mtimeMs)
    .map(({ bundlePath }) => bundlePath);
}

function resolveNamedBundle(bundleName) {
  const matches = listBundles(CONFIG.modelsDir).filter((bundlePath) => path.basename(bundlePath) === bundleName);
  if (matches.length === 1) {
    return matches[0];
  }
  if (matches.length > 1) {
    const relativeMatches = matches.map((bundlePath) => path.relative(process.cwd(), bundlePath));
    throw new Error(
      [
        `Multiple JS bundles match "${bundleName}".`,
        "Pass a full or relative path to the specific bundle instead:",
        ...relativeMatches.map((bundlePath) => `  ${bundlePath}`),
      ].join("\n")
    );
  }

  throw new Error(
    [
      `No JS bundle named "${bundleName}" was found.`,
      'Pass a full or relative path if the bundle is outside "models/".',
    ].join("\n")
  );
}

function resolveBundlePath(bundlePath) {
  if (bundlePath !== null) {
    const directPath = path.resolve(process.cwd(), bundlePath);
    if (fs.existsSync(directPath)) {
      return directPath;
    }
    if (path.isAbsolute(bundlePath) || bundlePath.includes("/") || bundlePath.includes("\\")) {
      throw new Error(`Bundle not found: ${directPath}`);
    }
    return resolveNamedBundle(bundlePath);
  }

  const bundles = listBundles(CONFIG.modelsDir);
  if (bundles.length === 0) {
    throw new Error(
      'No JS bundles were found in "models/". Save or resume a model first.'
    );
  }

  return bundles[0];
}

function loadBundle(bundlePath) {
  let bundle;
  try {
    bundle = fs.readFileSync(bundlePath);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Failed to read DreamPhraseGPT bundle "${bundlePath}": ${message}`);
  }

  if (bundle.length < CONFIG.bundleMagic.length + 4) {
    throw new Error(`Corrupt DreamPhraseGPT bundle: ${bundlePath}`);
  }

  const magic = bundle.subarray(0, CONFIG.bundleMagic.length).toString("ascii");

  if (magic !== CONFIG.bundleMagic) {
    throw new Error(`Invalid DreamPhraseGPT bundle: ${bundlePath}`);
  }

  const headerLength = bundle.readUInt32LE(CONFIG.bundleMagic.length);
  const headerStart = CONFIG.bundleMagic.length + 4;
  const headerEnd = headerStart + headerLength;

  if (headerEnd > bundle.length) {
    throw new Error(`Corrupt DreamPhraseGPT bundle header: ${bundlePath}`);
  }

  let header;
  try {
    header = JSON.parse(bundle.subarray(headerStart, headerEnd).toString("utf8"));
  } catch {
    throw new Error(`Corrupt DreamPhraseGPT bundle header JSON: ${bundlePath}`);
  }

  if (header.format !== CONFIG.bundleFormat) {
    throw new Error(`Unsupported bundle format: ${header.format ?? "<missing>"}`);
  }
  if (!Number.isInteger(header.version)) {
    throw new Error(`Bundle is missing a valid version: ${bundlePath}`);
  }
  if (header.version !== CONFIG.bundleVersion) {
    throw new Error(
      `Unsupported bundle version ${header.version}. Expected ${CONFIG.bundleVersion}.`
    );
  }

  const modelBytes = bundle.subarray(headerEnd);
  if (modelBytes.length === 0) {
    throw new Error(`Bundle is missing ONNX model bytes: ${bundlePath}`);
  }

  return {
    modelBytes,
    tokenizer: validateTokenizer(header.tokenizer, bundlePath),
    sourceFilter: validateSourceFilter(header.source_filter, bundlePath),
  };
}

function validateTokenizer(tokenizer, bundlePath) {
  if (tokenizer === null || typeof tokenizer !== "object") {
    throw new Error(`Bundle is missing tokenizer metadata: ${bundlePath}`);
  }

  const { id_to_char: idToChar, bos_id: bosId, block_size: blockSize, vocab_size: vocabSize } = tokenizer;

  if (!Array.isArray(idToChar) || !idToChar.every((value) => typeof value === "string")) {
    throw new Error(`Bundle tokenizer has an invalid id_to_char table: ${bundlePath}`);
  }

  if (!Number.isInteger(bosId) || !Number.isInteger(blockSize) || !Number.isInteger(vocabSize)) {
    throw new Error(`Bundle tokenizer metadata is incomplete: ${bundlePath}`);
  }

  if (blockSize <= 0 || vocabSize <= 0 || bosId < 0) {
    throw new Error(`Bundle tokenizer metadata is invalid: ${bundlePath}`);
  }

  if (bosId !== idToChar.length || vocabSize !== idToChar.length + 1) {
    throw new Error(`Bundle tokenizer metadata is inconsistent: ${bundlePath}`);
  }

  return { idToChar, bosId, blockSize, vocabSize };
}

function validateSourceFilter(sourceFilter, bundlePath) {
  if (sourceFilter == null) {
    return null;
  }
  if (typeof sourceFilter !== "object") {
    throw new Error(`Bundle source filter is invalid: ${bundlePath}`);
  }

  const {
    kind,
    version,
    bit_count: bitCount,
    hash_count: hashCount,
    bits_base64: bitsBase64,
    false_positive_rate: falsePositiveRate,
    item_count: itemCount,
  } = sourceFilter;

  if (kind !== CONFIG.sourceFilterKind || version !== CONFIG.sourceFilterVersion) {
    throw new Error(`Unsupported source filter payload in ${bundlePath}.`);
  }
  if (!Number.isInteger(bitCount) || bitCount <= 0) {
    throw new Error(`Bundle source filter has an invalid bit count: ${bundlePath}`);
  }
  if (!Number.isInteger(hashCount) || hashCount <= 0) {
    throw new Error(`Bundle source filter has an invalid hash count: ${bundlePath}`);
  }
  if (typeof bitsBase64 !== "string" || bitsBase64.length === 0) {
    throw new Error(`Bundle source filter is missing bit payload: ${bundlePath}`);
  }
  if (typeof falsePositiveRate !== "number" || !(falsePositiveRate > 0 && falsePositiveRate < 1)) {
    throw new Error(`Bundle source filter has an invalid false-positive rate: ${bundlePath}`);
  }
  if (!Number.isInteger(itemCount) || itemCount < 0) {
    throw new Error(`Bundle source filter has an invalid item count: ${bundlePath}`);
  }

  const bits = Buffer.from(bitsBase64, "base64");
  const expectedBytes = Math.ceil(bitCount / 8);
  if (bits.length !== expectedBytes) {
    throw new Error(`Bundle source filter has the wrong bit payload size: ${bundlePath}`);
  }

  return {
    bitCount,
    hashCount,
    bits,
    falsePositiveRate,
    itemCount,
  };
}

function softmax(logits, temperature) {
  const scaled = logits.map((value) => value / temperature);
  const max = Math.max(...scaled);
  const exps = scaled.map((value) => Math.exp(value - max));
  const sum = exps.reduce((left, right) => left + right, 0);
  return exps.map((value) => value / sum);
}

function sampleIndex(probabilities) {
  let remaining = Math.random();
  for (let index = 0; index < probabilities.length; index++) {
    remaining -= probabilities[index];
    if (remaining <= 0) {
      return index;
    }
  }
  return probabilities.length - 1;
}

function normalizeSourceText(text) {
  return text.trim();
}

function *iterHashIndices(text, sourceFilter) {
  const digest = crypto.createHash("sha256").update(Buffer.from(text, "utf8")).digest();
  const bitCount = BigInt(sourceFilter.bitCount);
  const first = digest.readBigUInt64LE(0);
  const secondRaw = digest.readBigUInt64LE(8);
  const second = secondRaw === 0n ? 0x9e3779b97f4a7c15n : secondRaw;
  for (let index = 0n; index < BigInt(sourceFilter.hashCount); index += 1n) {
    yield Number((first + (index * second)) % bitCount);
  }
}

function sourceFilterMatches(sourceFilter, text) {
  const normalized = normalizeSourceText(text);
  if (normalized.length === 0) {
    return false;
  }

  for (const index of iterHashIndices(normalized, sourceFilter)) {
    const byteIndex = Math.floor(index / 8);
    const bitOffset = index % 8;
    if ((sourceFilter.bits[byteIndex] & (1 << bitOffset)) === 0) {
      return false;
    }
  }

  return true;
}

function getLastLogits(logitsTensor, sequenceLength, expectedVocabSize) {
  if (logitsTensor === null || typeof logitsTensor !== "object" || !Array.isArray(logitsTensor.dims)) {
    throw new Error('ONNX session returned an invalid "logits" tensor.');
  }
  if (logitsTensor.data === null || typeof logitsTensor.data !== "object" || typeof logitsTensor.data.length !== "number") {
    throw new Error('ONNX session returned a "logits" tensor without readable data.');
  }

  const [batchSize, outputSequenceLength, vocabSize] = logitsTensor.dims;
  if (logitsTensor.dims.length !== 3) {
    throw new Error(
      `Unexpected ONNX logits rank ${logitsTensor.dims.length}. Expected 3 dimensions.`
    );
  }
  if (batchSize !== 1) {
    throw new Error(`Unexpected ONNX batch size ${batchSize}. Expected 1.`);
  }
  if (outputSequenceLength !== sequenceLength) {
    throw new Error(
      `Unexpected ONNX sequence length ${outputSequenceLength}. Expected ${sequenceLength}.`
    );
  }
  if (vocabSize !== expectedVocabSize) {
    throw new Error(`Unexpected ONNX vocab size ${vocabSize}. Expected ${expectedVocabSize}.`);
  }
  if (logitsTensor.data.length < sequenceLength * vocabSize) {
    throw new Error('ONNX session returned an incomplete "logits" tensor.');
  }

  const offset = (sequenceLength - 1) * vocabSize;
  return Array.from(logitsTensor.data.slice(offset, offset + vocabSize));
}

async function generateOneSample(session, tokenizer, temperature) {
  const { idToChar, bosId, blockSize, vocabSize } = tokenizer;
  const tokenIds = [bosId];
  const characters = [];

  for (let step = 0; step < blockSize; step++) {
    const window = tokenIds.slice(-blockSize);
    const input = new ort.Tensor("int64", BigInt64Array.from(window.map(BigInt)), [1, window.length]);
    const output = await session.run({ idx: input });
    if (
      output === null ||
      typeof output !== "object" ||
      !Object.prototype.hasOwnProperty.call(output, "logits")
    ) {
      throw new Error('ONNX session output is missing "logits".');
    }
    const logits = getLastLogits(output.logits, window.length, vocabSize);
    const probabilities = softmax(logits, temperature);
    const tokenId = sampleIndex(probabilities);

    if (tokenId === bosId) {
      break;
    }

    characters.push(idToChar[tokenId]);
    tokenIds.push(tokenId);
  }

  return characters.join("");
}

async function generateFilteredSamples(session, tokenizer, sourceFilter, options) {
  if (sourceFilter === null) {
    throw new Error(
      "This bundle does not include source filter metadata. Regenerate it with the current DreamPhraseGPT version before generating."
    );
  }

  const samples = [];
  for (let index = 0; index < options.samples; index++) {
    let accepted = null;
    for (let attempt = 0; attempt < CONFIG.sourceFilterMaxRetries; attempt++) {
      const candidate = await generateOneSample(session, tokenizer, options.temperature);
      if (!sourceFilterMatches(sourceFilter, candidate)) {
        accepted = candidate;
        break;
      }
    }

    if (accepted === null) {
      throw new Error(
        `Failed to sample a non-source line within ${CONFIG.sourceFilterMaxRetries} attempts. Increase --temperature or train for fewer steps.`
      );
    }

    samples.push(accepted);
  }
  return samples;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printUsage();
    return;
  }

  const bundlePath = resolveBundlePath(options.bundlePath);
  const { modelBytes, tokenizer, sourceFilter } = loadBundle(bundlePath);
  const session = await ort.InferenceSession.create(modelBytes);
  const samples = await generateFilteredSamples(session, tokenizer, sourceFilter, options);
  samples.forEach((sample) => console.log(sample));
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
