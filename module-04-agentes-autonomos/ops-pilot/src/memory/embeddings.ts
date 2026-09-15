import { pipeline, type FeatureExtractionPipeline } from "@huggingface/transformers";

const MODEL_ID = "onnx-community/all-MiniLM-L6-v2-ONNX";
const CACHE_DIR = "./.cache/transformers";

let extractorPromise: Promise<FeatureExtractionPipeline> | undefined;

function getExtractor(): Promise<FeatureExtractionPipeline> {
  if (!extractorPromise) {
    extractorPromise = pipeline("feature-extraction", MODEL_ID, { cache_dir: CACHE_DIR });
  }
  return extractorPromise;
}

/**
 * Embedding local de `text` — 384 posições, já normalizado (`pooling: "mean", normalize: true`,
 * research.md item 1). O modelo (`onnx-community/all-MiniLM-L6-v2-ONNX`) só é carregado no
 * primeiro `embed()` chamado; importar este módulo não baixa nem carrega nada.
 */
export async function embed(text: string): Promise<Float32Array> {
  const extractor = await getExtractor();
  const output = await extractor(text, { pooling: "mean", normalize: true });
  return output.data as Float32Array;
}
