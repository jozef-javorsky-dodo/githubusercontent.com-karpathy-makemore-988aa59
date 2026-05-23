import * as tf from "@tensorflow/tfjs-node";
import * as fs from "fs";
import * as https from "https";
import * as path from "path";

tf.enableProdMode();

interface ModelConfig {
  vocabSize: number;
  blockSize: number;
  embedDim: number;
  numHeads: number;
  numLayers: number;
}

interface TrainConfig {
  batchSize: number;
  epochs: number;
  learningRate: number;
}

const APP_CONFIG = {
  model: {
    vocabSize: 0,
    blockSize: 16,
    embedDim: 64,
    numHeads: 4,
    numLayers: 2,
  } as ModelConfig,
  train: {
    batchSize: 256,
    epochs: 10,
    learningRate: 0.001,
  } as TrainConfig,
  inference: {
    temperature: 0.8,
    maxSteps: 20,
    numSamples: 10,
  },
  paths: {
    modelDir: path.join(__dirname, "saved_gpt_weights"),
    dataUrl:
      "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt",
    dataFile: path.join(__dirname, "input.txt"),
  },
};

class Logger {
  static info(msg: string) {
    console.log(`[INFO] ${msg}`);
  }
  static error(msg: string, err?: any) {
    console.error(`[ERROR] ${msg}`, err);
  }
  static progress(
    epoch: number,
    totalEpochs: number,
    batch: number,
    totalBatches: number,
    loss: number,
  ) {
    process.stdout.write(
      `\r[TRAIN] Epoch ${epoch}/${totalEpochs} | Batch ${batch}/${totalBatches} | Loss: ${loss.toFixed(4)}`,
    );
  }
}

class DataFetcher {
  static async download(url: string, dest: string): Promise<void> {
    if (fs.existsSync(dest)) return;
    Logger.info(`Downloading dataset from ${url}...`);
    return new Promise((resolve, reject) => {
      const file = fs.createWriteStream(dest);
      https
        .get(url, (res) => {
          if (res.statusCode !== 200)
            return reject(new Error(`Failed to fetch: ${res.statusCode}`));
          res.pipe(file);
          file.on("finish", () => {
            file.close();
            resolve();
          });
        })
        .on("error", (err) => {
          fs.unlink(dest, () => reject(err));
        });
    });
  }
}

class Tokenizer {
  private char2id = new Map<string, number>();
  public id2char = new Map<number, string>();
  public vocabSize = 0;
  public BOS = 0;

  public build(text: string): void {
    const chars = Array.from(new Set(text)).sort();
    this.vocabSize = chars.length + 1;
    this.BOS = chars.length;

    chars.forEach((ch, i) => {
      this.char2id.set(ch, i);
      this.id2char.set(i, ch);
    });
    this.id2char.set(this.BOS, "");
  }

  public encode(str: string): number[] {
    return str.split("").map((ch) => this.char2id.get(ch) ?? this.BOS);
  }
}

class DataLoader {
  private indices: Uint32Array;
  public numBatches: number;
  private xFlat: Int32Array;
  private yFlat: Int32Array;

  constructor(
    private tokens: Int32Array,
    private blockSize: number,
    private batchSize: number,
  ) {
    const numSamples = tokens.length - blockSize;
    this.indices = new Uint32Array(numSamples);
    for (let i = 0; i < numSamples; i++) this.indices[i] = i;

    this.numBatches = Math.floor(numSamples / this.batchSize);

    this.xFlat = new Int32Array(this.batchSize * this.blockSize);
    this.yFlat = new Int32Array(this.batchSize * this.blockSize);

    this.shuffle();
  }

  public shuffle() {
    for (let i = this.indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [this.indices[i], this.indices[j]] = [this.indices[j], this.indices[i]];
    }
  }

  public getBatch(batchIdx: number): { x: tf.Tensor2D; y: tf.Tensor2D } {
    const start = batchIdx * this.batchSize;

    for (let i = 0; i < this.batchSize; i++) {
      const tokenIdx = this.indices[start + i];
      this.xFlat.set(
        this.tokens.subarray(tokenIdx, tokenIdx + this.blockSize),
        i * this.blockSize,
      );
      this.yFlat.set(
        this.tokens.subarray(tokenIdx + 1, tokenIdx + this.blockSize + 1),
        i * this.blockSize,
      );
    }

    return {
      x: tf.tensor2d(this.xFlat, [this.batchSize, this.blockSize], "int32"),
      y: tf.tensor2d(this.yFlat, [this.batchSize, this.blockSize], "int32"),
    };
  }
}

abstract class Module {
  private _weights: tf.Variable[] = [];
  private _submodules: Module[] = [];
  private _buffers: tf.Tensor[] = [];

  protected registerWeight(shape: number[], initVal?: number): tf.Variable {
    const init =
      initVal !== undefined
        ? tf.fill(shape, initVal)
        : tf.randomNormal(shape, 0, 0.02);
    const v = tf.variable(init);
    this._weights.push(v);
    return v;
  }

  protected registerBuffer(tensor: tf.Tensor): tf.Tensor {
    this._buffers.push(tensor);
    return tensor;
  }

  protected registerModule<T extends Module>(module: T): T {
    this._submodules.push(module);
    return module;
  }

  public getParameters(): tf.Variable[] {
    let params = [...this._weights];
    for (const mod of this._submodules)
      params = params.concat(mod.getParameters());
    return params;
  }

  public dispose() {
    this._weights.forEach((w) => w.dispose());
    this._buffers.forEach((b) => b.dispose());
    this._submodules.forEach((m) => m.dispose());
  }

  abstract forward(...args: any[]): tf.Tensor;
}

class Linear extends Module {
  private w: tf.Variable;
  private b: tf.Variable | null;

  constructor(
    inFeatures: number,
    outFeatures: number,
    useBias: boolean = true,
  ) {
    super();
    this.w = this.registerWeight([inFeatures, outFeatures]);
    this.b = useBias ? this.registerWeight([outFeatures], 0) : null;
  }

  forward(x: tf.Tensor): tf.Tensor {
    const [B, T, C] = x.shape;
    const x2d = x.reshape([-1, C]);
    let out2d = tf.matMul(x2d, this.w);
    if (this.b) out2d = out2d.add(this.b);
    return out2d.reshape([B, T, -1]);
  }
}

class LayerNorm extends Module {
  private g: tf.Variable;
  private b: tf.Variable;

  constructor(features: number) {
    super();
    this.g = this.registerWeight([features], 1);
    this.b = this.registerWeight([features], 0);
  }

  forward(x: tf.Tensor): tf.Tensor {
    const moments = tf.moments(x, -1, true);
    return x
      .sub(moments.mean)
      .div(tf.sqrt(moments.variance.add(1e-5)))
      .mul(this.g)
      .add(this.b);
  }
}

class CausalSelfAttention extends Module {
  private c_attn: Linear;
  private c_proj: Linear;
  private mask: tf.Tensor;
  private headDim: number;

  constructor(private config: ModelConfig) {
    super();
    this.c_attn = this.registerModule(
      new Linear(config.embedDim, 3 * config.embedDim),
    );
    this.c_proj = this.registerModule(
      new Linear(config.embedDim, config.embedDim),
    );
    this.headDim = config.embedDim / config.numHeads;

    const indices = tf.range(0, config.blockSize, 1, "int32");
    const m = indices
      .reshape([config.blockSize, 1])
      .greaterEqual(indices.reshape([1, config.blockSize]))
      .cast("float32");
    this.mask = this.registerBuffer(m.sub(1).mul(1e9));
  }

  forward(x: tf.Tensor): tf.Tensor {
    const [B, T, C] = x.shape;

    let qkv = this.c_attn.forward(x);
    let [q, k, v] = tf.split(qkv, 3, -1);

    const reshapeT = (t: tf.Tensor) =>
      t
        .reshape([B, T, this.config.numHeads, this.headDim])
        .transpose([0, 2, 1, 3]);
    let Q = reshapeT(q);
    let K = reshapeT(k);
    let V = reshapeT(v);

    let att = tf.matMul(Q, K, false, true).div(Math.sqrt(this.headDim));

    const currentMask = this.mask.slice([0, 0], [T, T]);
    att = att.add(currentMask);
    att = tf.softmax(att, -1);

    let out = tf.matMul(att, V);
    out = out.transpose([0, 2, 1, 3]).reshape([B, T, C]);
    return this.c_proj.forward(out);
  }
}

class MLP extends Module {
  private c_fc: Linear;
  private c_proj: Linear;

  constructor(config: ModelConfig) {
    super();
    this.c_fc = this.registerModule(
      new Linear(config.embedDim, 4 * config.embedDim),
    );
    this.c_proj = this.registerModule(
      new Linear(4 * config.embedDim, config.embedDim),
    );
  }

  forward(x: tf.Tensor): tf.Tensor {
    let h = this.c_fc.forward(x);
    h = tf.relu(h);
    return this.c_proj.forward(h);
  }
}

class TransformerBlock extends Module {
  private ln_1: LayerNorm;
  private attn: CausalSelfAttention;
  private ln_2: LayerNorm;
  private mlp: MLP;

  constructor(config: ModelConfig) {
    super();
    this.ln_1 = this.registerModule(new LayerNorm(config.embedDim));
    this.attn = this.registerModule(new CausalSelfAttention(config));
    this.ln_2 = this.registerModule(new LayerNorm(config.embedDim));
    this.mlp = this.registerModule(new MLP(config));
  }

  forward(x: tf.Tensor): tf.Tensor {
    x = x.add(this.attn.forward(this.ln_1.forward(x)));
    x = x.add(this.mlp.forward(this.ln_2.forward(x)));
    return x;
  }
}

class GPT extends Module {
  private wte: tf.Variable;
  private wpe: tf.Variable;
  private blocks: TransformerBlock[] = [];
  private ln_f: LayerNorm;
  private lm_head: Linear;

  constructor(public config: ModelConfig) {
    super();
    this.wte = this.registerWeight([config.vocabSize, config.embedDim]);
    this.wpe = this.registerWeight([config.blockSize, config.embedDim]);

    for (let i = 0; i < config.numLayers; i++) {
      this.blocks.push(this.registerModule(new TransformerBlock(config)));
    }

    this.ln_f = this.registerModule(new LayerNorm(config.embedDim));
    this.lm_head = this.registerModule(
      new Linear(config.embedDim, config.vocabSize, false),
    );
  }

  forward(idx: tf.Tensor): tf.Tensor {
    const [B, T] = idx.shape;

    let x = tf.gather(this.wte, idx.cast("int32"));
    const pos = tf.range(0, T, 1, "int32");
    const posEmb = tf.gather(this.wpe, pos);
    x = x.add(posEmb);

    for (const block of this.blocks) x = block.forward(x);

    x = this.ln_f.forward(x);
    return this.lm_head.forward(x);
  }

  public save(dir: string) {
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    const manifest: any = {};
    this.getParameters().forEach((w, i) => {
      const buffer = Buffer.from(w.dataSync().buffer);
      fs.writeFileSync(path.join(dir, `w_${i}.bin`), buffer);
      manifest[`w_${i}`] = { shape: w.shape, file: `w_${i}.bin` };
    });
    fs.writeFileSync(path.join(dir, "manifest.json"), JSON.stringify(manifest));
    Logger.info(`Model saved to ${dir}`);
  }

  public load(dir: string) {
    const manifest = JSON.parse(
      fs.readFileSync(path.join(dir, "manifest.json"), "utf-8"),
    );
    this.getParameters().forEach((w, i) => {
      const info = manifest[`w_${i}`];
      const buffer = fs.readFileSync(path.join(dir, info.file));
      const floatArray = new Float32Array(
        buffer.buffer,
        buffer.byteOffset,
        buffer.byteLength / 4,
      );

      const tempTensor = tf.tensor(floatArray, info.shape);
      w.assign(tempTensor);
      tempTensor.dispose();
    });
    Logger.info(`Model loaded from ${dir}`);
  }
}

class Trainer {
  constructor(
    private model: GPT,
    private config: TrainConfig,
  ) {}

  public async train(dataLoader: DataLoader) {
    const optimizer = tf.train.adam(this.config.learningRate);

    for (let epoch = 1; epoch <= this.config.epochs; epoch++) {
      let epochLoss = 0;
      dataLoader.shuffle();

      for (let b = 0; b < dataLoader.numBatches; b++) {
        const { x, y } = dataLoader.getBatch(b);

        const loss = tf.tidy(() => {
          const { value, grads } = optimizer.computeGradients(() => {
            const logits = this.model.forward(x);
            const targetsOneHot = tf.oneHot(
              y.flatten(),
              this.model.config.vocabSize,
            );
            const logitsFlat = logits.reshape([
              -1,
              this.model.config.vocabSize,
            ]);
            return tf.losses.softmaxCrossEntropy(targetsOneHot, logitsFlat);
          });

          optimizer.applyGradients(grads);
          return value;
        });

        const lossVal = (await loss.data())[0];
        epochLoss += lossVal;

        tf.dispose([x, y, loss]);

        Logger.progress(
          epoch,
          this.config.epochs,
          b + 1,
          dataLoader.numBatches,
          epochLoss / (b + 1),
        );
      }
      console.log();
    }
  }
}

class TextGenerator {
  constructor(
    private model: GPT,
    private tokenizer: Tokenizer,
  ) {}

  public async generate(
    temperature: number,
    maxSteps: number,
  ): Promise<string> {
    let currentSeq = [this.tokenizer.BOS];
    let generated = "";

    for (let step = 0; step < maxSteps; step++) {
      const sampleTensor = tf.tidy(() => {
        const context = currentSeq.slice(-this.model.config.blockSize);
        const inputTensor = tf.tensor2d(
          [context],
          [1, context.length],
          "int32",
        );

        const logits = this.model.forward(inputTensor);
        const lastLogits = logits
          .slice([0, context.length - 1, 0], [1, 1, -1])
          .squeeze();

        const scaledLogits = lastLogits.div(tf.scalar(temperature));
        return tf.multinomial(scaledLogits as tf.Tensor1D, 1);
      });

      const nextTokenId = await sampleTensor.data();
      sampleTensor.dispose();

      const id = nextTokenId[0];
      if (id === this.tokenizer.BOS) break;
      generated += this.tokenizer.id2char.get(id);
      currentSeq.push(id);
    }
    return generated;
  }
}

async function main() {
  Logger.info("Initializing GPT Pipeline...");

  await DataFetcher.download(
    APP_CONFIG.paths.dataUrl,
    APP_CONFIG.paths.dataFile,
  );
  const text = fs.readFileSync(APP_CONFIG.paths.dataFile, "utf-8");

  const tokenizer = new Tokenizer();
  tokenizer.build(text);
  APP_CONFIG.model.vocabSize = tokenizer.vocabSize;

  const model = new GPT(APP_CONFIG.model);

  if (fs.existsSync(path.join(APP_CONFIG.paths.modelDir, "manifest.json"))) {
    model.load(APP_CONFIG.paths.modelDir);
  } else {
    Logger.info("Preparing dataset for training...");

    const rawTokens = text
      .split("\n")
      .map((l) => l.trim())
      .filter((l) => l.length > 0);
    const flatTokens: number[] = [];
    rawTokens.forEach((doc) =>
      flatTokens.push(tokenizer.BOS, ...tokenizer.encode(doc), tokenizer.BOS),
    );

    const dataLoader = new DataLoader(
      new Int32Array(flatTokens),
      APP_CONFIG.model.blockSize,
      APP_CONFIG.train.batchSize,
    );
    Logger.info(`Training on ${dataLoader.numBatches} batches per epoch...`);

    const trainer = new Trainer(model, APP_CONFIG.train);
    await trainer.train(dataLoader);

    model.save(APP_CONFIG.paths.modelDir);
  }

  Logger.info(
    `\n--- Generating Names (Temperature: ${APP_CONFIG.inference.temperature}) ---`,
  );
  const generator = new TextGenerator(model, tokenizer);

  for (let i = 0; i < APP_CONFIG.inference.numSamples; i++) {
    const sample = await generator.generate(
      APP_CONFIG.inference.temperature,
      APP_CONFIG.inference.maxSteps,
    );
    console.log(`Sample ${i + 1}: ${sample}`);
  }

  model.dispose();
}

main().catch((err) => Logger.error("Application failed", err));
