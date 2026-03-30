import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import * as https from 'https';
import * as path from 'path';

tf.enableProdMode(); // Disables debug validations for maximum performance

// ============================================================================
// 1. CONFIGURATION
// ============================================================================
const CONFIG = {
    blockSize: 16,        // Maximum context length
    embedDim: 64,         // Embedding dimension
    numHeads: 4,          // Number of attention heads
    numLayers: 2,         // Number of Transformer blocks
    batchSize: 256,       // Sequences processed in parallel
    epochs: 10,           // Full passes over the dataset
    learningRate: 0.001,  // Adam learning rate
    temperature: 0.8,     // Inference creativity
    modelDir: path.join(__dirname, 'saved_gpt_weights'),
    dataUrl: 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt',
    dataFile: path.join(__dirname, 'input.txt')
};

// ============================================================================
// 2. DATA PROCESSING
// ============================================================================
async function downloadData(url: string, dest: string): Promise<void> {
    if (fs.existsSync(dest)) return;
    console.log("Downloading dataset...");
    return new Promise((resolve, reject) => {
        const file = fs.createWriteStream(dest);
        https.get(url, (res) => {
            res.pipe(file);
            file.on('finish', () => { file.close(); resolve(); });
        }).on('error', (err) => fs.unlink(dest, () => reject(err)));
    });
}

class Tokenizer {
    public char2id: Map<string, number> = new Map();
    public id2char: Map<number, string> = new Map();
    public vocabSize: number = 0;
    public BOS: number = 0;

    build(text: string) {
        const chars = Array.from(new Set(text)).sort();
        this.vocabSize = chars.length + 1;
        this.BOS = chars.length;
        chars.forEach((ch, i) => {
            this.char2id.set(ch, i);
            this.id2char.set(i, ch);
        });
        this.id2char.set(this.BOS, '');
    }

    encode(str: string): number[] {
        return str.split('').map(ch => this.char2id.get(ch) ?? this.BOS);
    }
}

// ============================================================================
// 3. MODEL ARCHITECTURE (TFJS Core Ops)
// ============================================================================
class GPT {
    public weights: tf.Variable[] = [];
    private wte: tf.Variable;
    private wpe: tf.Variable;
    private blocks: any[] = [];
    private ln_f_g: tf.Variable;
    private ln_f_b: tf.Variable;
    private lm_head: tf.Variable;

    constructor(public vocabSize: number) {
        this.wte = this.addVar([vocabSize, CONFIG.embedDim]);
        this.wpe = this.addVar([CONFIG.blockSize, CONFIG.embedDim]);

        for (let i = 0; i < CONFIG.numLayers; i++) {
            this.blocks.push({
                ln1_g: this.addVar([CONFIG.embedDim], 1),
                ln1_b: this.addVar([CONFIG.embedDim], 0),
                attn_w: this.addVar([CONFIG.embedDim, 3 * CONFIG.embedDim]),
                attn_b: this.addVar([3 * CONFIG.embedDim], 0),
                proj_w: this.addVar([CONFIG.embedDim, CONFIG.embedDim]),
                proj_b: this.addVar([CONFIG.embedDim], 0),
                ln2_g: this.addVar([CONFIG.embedDim], 1),
                ln2_b: this.addVar([CONFIG.embedDim], 0),
                mlp_fc_w: this.addVar([CONFIG.embedDim, 4 * CONFIG.embedDim]),
                mlp_fc_b: this.addVar([4 * CONFIG.embedDim], 0),
                mlp_proj_w: this.addVar([4 * CONFIG.embedDim, CONFIG.embedDim]),
                mlp_proj_b: this.addVar([CONFIG.embedDim], 0),
            });
        }

        this.ln_f_g = this.addVar([CONFIG.embedDim], 1);
        this.ln_f_b = this.addVar([CONFIG.embedDim], 0);
        this.lm_head = this.addVar([CONFIG.embedDim, vocabSize]);
    }

    private addVar(shape: number[], initVal?: number): tf.Variable {
        const init = initVal !== undefined ? tf.fill(shape, initVal) : tf.randomNormal(shape, 0, 0.02);
        const v = tf.variable(init);
        this.weights.push(v);
        return v;
    }

    private layerNorm(x: tf.Tensor, g: tf.Variable, b: tf.Variable): tf.Tensor {
        const moments = tf.moments(x, -1, true);
        return x.sub(moments.mean).div(tf.sqrt(moments.variance.add(1e-5))).mul(g).add(b);
    }

    // Safely applies a 2D weight matrix to a 3D tensor
    private dense(x: tf.Tensor, w: tf.Variable, b: tf.Variable): tf.Tensor {
        const [B, T, C] = x.shape;
        const x2d = x.reshape([-1, C]);
        const out2d = tf.matMul(x2d, w).add(b);
        return out2d.reshape([B, T, -1]);
    }

    public forward(idx: tf.Tensor): tf.Tensor {
        const [B, T] = idx.shape;
        
        let x = tf.gather(this.wte, idx.cast('int32'));
        const pos = tf.range(0, T, 1, 'int32');
        const posEmb = tf.gather(this.wpe, pos);
        x = x.add(posEmb);

        // Causal Mask: 0 for past, -10000 for future
        const indices = tf.range(0, T, 1, 'int32');
        const mask = indices.reshape([T, 1]).greaterEqual(indices.reshape([1, T])).cast('float32');
        const additiveMask = mask.sub(1).mul(1e4); 

        for (const block of this.blocks) {
            // 1. Attention
            let norm1 = this.layerNorm(x, block.ln1_g, block.ln1_b);
            let qkv = this.dense(norm1, block.attn_w, block.attn_b);
            let [q, k, v] = tf.split(qkv, 3, -1);
            
            const headDim = CONFIG.embedDim / CONFIG.numHeads;
            const reshapeT = (t: tf.Tensor) => t.reshape([B, T, CONFIG.numHeads, headDim]).transpose([0, 2, 1, 3]);
            
            let Q = reshapeT(q);
            let K = reshapeT(k);
            let V = reshapeT(v);

            let att = tf.matMul(Q, K, false, true).div(Math.sqrt(headDim));
            att = att.add(additiveMask);
            att = tf.softmax(att, -1);

            let out = tf.matMul(att, V);
            out = out.transpose([0, 2, 1, 3]).reshape([B, T, CONFIG.embedDim]);
            out = this.dense(out, block.proj_w, block.proj_b);
            x = x.add(out);

            // 2. MLP
            let norm2 = this.layerNorm(x, block.ln2_g, block.ln2_b);
            let mlp = this.dense(norm2, block.mlp_fc_w, block.mlp_fc_b);
            mlp = tf.relu(mlp);
            mlp = this.dense(mlp, block.mlp_proj_w, block.mlp_proj_b);
            x = x.add(mlp);
        }

        x = this.layerNorm(x, this.ln_f_g, this.ln_f_b);
        return this.dense(x, this.lm_head, tf.variable(tf.zeros([this.vocabSize])));
    }

    // Production-ready binary serialization
    public save(dir: string) {
        if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
        const manifest: any = {};
        this.weights.forEach((w, i) => {
            const data = w.dataSync(); // Float32Array
            const buffer = Buffer.from(data.buffer);
            fs.writeFileSync(path.join(dir, `w_${i}.bin`), buffer);
            manifest[`w_${i}`] = { shape: w.shape, file: `w_${i}.bin` };
        });
        fs.writeFileSync(path.join(dir, 'manifest.json'), JSON.stringify(manifest));
    }

    public load(dir: string) {
        const manifest = JSON.parse(fs.readFileSync(path.join(dir, 'manifest.json'), 'utf-8'));
        this.weights.forEach((w, i) => {
            const info = manifest[`w_${i}`];
            const buffer = fs.readFileSync(path.join(dir, info.file));
            const floatArray = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
            w.assign(tf.tensor(floatArray, info.shape));
        });
    }
}

// ============================================================================
// 4. TRAINING & INFERENCE
// ============================================================================
async function main() {
    await downloadData(CONFIG.dataUrl, CONFIG.dataFile);
    const text = fs.readFileSync(CONFIG.dataFile, 'utf-8');
    
    const tokenizer = new Tokenizer();
    tokenizer.build(text);

    const model = new GPT(tokenizer.vocabSize);

    if (fs.existsSync(path.join(CONFIG.modelDir, 'manifest.json'))) {
        console.log("Loading existing model from disk...");
        model.load(CONFIG.modelDir);
    } else {
        console.log("Preparing dataset for training...");
        const allTokens: number[] = [];
        text.split('\n').map(l => l.trim()).filter(l => l.length > 0).forEach(doc => {
            allTokens.push(tokenizer.BOS, ...tokenizer.encode(doc), tokenizer.BOS);
        });

        const xData: number[][] = [];
        const yData: number[][] = [];
        // Stride of 1 creates overlapping sequences, maximizing training data
        for (let i = 0; i < allTokens.length - CONFIG.blockSize; i++) {
            xData.push(allTokens.slice(i, i + CONFIG.blockSize));
            yData.push(allTokens.slice(i + 1, i + CONFIG.blockSize + 1));
        }

        console.log(`Training on ${xData.length} sequences...`);
        const optimizer = tf.train.adam(CONFIG.learningRate);
        const numBatches = Math.floor(xData.length / CONFIG.batchSize);

        for (let epoch = 0; epoch < CONFIG.epochs; epoch++) {
            let epochLoss = 0;
            for (let b = 0; b < numBatches; b++) {
                const loss = tf.tidy(() => {
                    const xBatch = tf.tensor2d(xData.slice(b * CONFIG.batchSize, (b + 1) * CONFIG.batchSize));
                    const yBatch = tf.tensor2d(yData.slice(b * CONFIG.batchSize, (b + 1) * CONFIG.batchSize), undefined, 'int32');

                    const { value, grads } = optimizer.computeGradients(() => {
                        const logits = model.forward(xBatch);
                        const targetsOneHot = tf.oneHot(yBatch.flatten(), tokenizer.vocabSize);
                        const logitsFlat = logits.reshape([-1, tokenizer.vocabSize]);
                        return tf.losses.softmaxCrossEntropy(targetsOneHot, logitsFlat);
                    });
                    
                    optimizer.applyGradients(grads);
                    tf.dispose(grads); // CRITICAL: Prevents massive memory leak
                    return value;
                });

                epochLoss += loss.dataSync()[0];
                loss.dispose(); // CRITICAL: Dispose scalar loss
                process.stdout.write(`Epoch ${epoch + 1}/${CONFIG.epochs} | Batch ${b + 1}/${numBatches} | Loss: ${(epochLoss / (b + 1)).toFixed(4)}\r`);
            }
            console.log(); 
        }
        console.log("Saving model...");
        model.save(CONFIG.modelDir);
    }

    console.log(`\n--- Generating Names (Temperature: ${CONFIG.temperature}) ---`);
    for (let i = 0; i < 15; i++) {
        let currentSeq = [tokenizer.BOS];
        let generated = "";

        for (let step = 0; step < 20; step++) {
            const nextTokenId = tf.tidy(() => {
                // Do not left-pad with BOS. Pass exact sequence length up to blockSize.
                const context = currentSeq.slice(-CONFIG.blockSize);
                const inputTensor = tf.tensor2d([context], [1, context.length]);
                
                const logits = model.forward(inputTensor);
                const lastLogits = logits.slice([0, context.length - 1, 0], [1, 1, -1]).squeeze();
                
                // tf.multinomial requires unnormalized logits, NOT probabilities
                const scaledLogits = lastLogits.div(tf.scalar(CONFIG.temperature));
                return tf.multinomial(scaledLogits as tf.Tensor1D, 1).dataSync()[0];
            });

            if (nextTokenId === tokenizer.BOS) break;
            generated += tokenizer.id2char.get(nextTokenId);
            currentSeq.push(nextTokenId);
        }
        console.log(`Sample ${i + 1}: ${generated}`);
    }
}

main().catch(console.error);