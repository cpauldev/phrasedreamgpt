/**
 * run_js_bundle.js — run a PhraseDreamGPT JS bundle exported by `phrasedreamgpt.py`
 *
 * Usage:
 *   node run_js_bundle.js
 *   node run_js_bundle.js english_names.model
 *   node run_js_bundle.js english_names.model --samples 30 --temperature 0.7
 *
 * If no bundle path is provided, the newest `*.model` file anywhere in `models/` is used.
 */

const fs = require("fs");
const path = require("path");
const ort = require("onnxruntime-node");

const CONFIG = Object.freeze({
  modelsDir: path.resolve(__dirname, "models"),
  defaultSamples: 20,
  defaultTemperature: 0.8,
  bundleMagic: "PDBGONNX",
  bundleFormat: "phrasedreamgpt-onnx-bundle",
});

const USAGE_TEXT = [
  "Usage:",
  "  node run_js_bundle.js [bundle-path] [--samples N] [--temperature FLOAT]",
  "",
  "Examples:",
  "  node run_js_bundle.js",
  "  node run_js_bundle.js english_names.model",
  "  node run_js_bundle.js english_names.model --samples 40 --temperature 0.7",
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

  return collectBundlePaths(modelsDir).sort((left, right) => fs.statSync(right).mtimeMs - fs.statSync(left).mtimeMs);
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
  return path.resolve(process.cwd(), bundleName);
}

function resolveBundlePath(bundlePath) {
  if (bundlePath !== null) {
    const directPath = path.resolve(process.cwd(), bundlePath);
    if (fs.existsSync(directPath)) {
      return directPath;
    }
    if (path.isAbsolute(bundlePath) || bundlePath.includes("/") || bundlePath.includes("\\")) {
      return directPath;
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
  const bundle = fs.readFileSync(bundlePath);
  if (bundle.length < CONFIG.bundleMagic.length + 4) {
    throw new Error(`Corrupt PhraseDreamGPT bundle: ${bundlePath}`);
  }

  const magic = bundle.subarray(0, CONFIG.bundleMagic.length).toString("ascii");

  if (magic !== CONFIG.bundleMagic) {
    throw new Error(`Invalid PhraseDreamGPT bundle: ${bundlePath}`);
  }

  const headerLength = bundle.readUInt32LE(CONFIG.bundleMagic.length);
  const headerStart = CONFIG.bundleMagic.length + 4;
  const headerEnd = headerStart + headerLength;

  if (headerEnd > bundle.length) {
    throw new Error(`Corrupt PhraseDreamGPT bundle header: ${bundlePath}`);
  }

  const header = JSON.parse(bundle.subarray(headerStart, headerEnd).toString("utf8"));
  if (header.format !== CONFIG.bundleFormat) {
    throw new Error(`Unsupported bundle format: ${header.format ?? "<missing>"}`);
  }

  const modelBytes = bundle.subarray(headerEnd);
  if (modelBytes.length === 0) {
    throw new Error(`Bundle is missing ONNX model bytes: ${bundlePath}`);
  }

  return {
    modelBytes,
    tokenizer: validateTokenizer(header.tokenizer, bundlePath),
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

function getLastLogits(logitsTensor, sequenceLength, expectedVocabSize) {
  const vocabSize = logitsTensor.dims.at(-1);
  if (vocabSize !== expectedVocabSize) {
    throw new Error(`Unexpected ONNX vocab size ${vocabSize}. Expected ${expectedVocabSize}.`);
  }

  const offset = (sequenceLength - 1) * vocabSize;
  return Array.from(logitsTensor.data.slice(offset, offset + vocabSize));
}

async function generateOneName(session, tokenizer, temperature) {
  const { idToChar, bosId, blockSize, vocabSize } = tokenizer;
  const tokenIds = [bosId];
  const characters = [];

  for (let step = 0; step < blockSize; step++) {
    const window = tokenIds.slice(-blockSize);
    const input = new ort.Tensor("int64", BigInt64Array.from(window.map(BigInt)), [1, window.length]);
    const output = await session.run({ idx: input });
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

async function generateNames(session, tokenizer, options) {
  const names = [];
  for (let index = 0; index < options.samples; index++) {
    names.push(await generateOneName(session, tokenizer, options.temperature));
  }
  return names;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printUsage();
    return;
  }

  const bundlePath = resolveBundlePath(options.bundlePath);
  const { modelBytes, tokenizer } = loadBundle(bundlePath);
  const session = await ort.InferenceSession.create(modelBytes);
  const names = await generateNames(session, tokenizer, options);
  names.forEach((name) => console.log(name));
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
