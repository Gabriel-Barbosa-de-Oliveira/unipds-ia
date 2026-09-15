export interface ScoredMatch<T> {
  readonly item: T;
  readonly score: number;
}

interface Candidate<T> {
  readonly item: T;
  readonly embedding: Float32Array;
}

/** Produto escalar entre dois vetores — similaridade de significado quando ambos já vêm normalizados (research.md item 2). */
export function dotProduct(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    sum += (a[i] ?? 0) * (b[i] ?? 0);
  }
  return sum;
}

/**
 * Ordena `candidates` por similaridade a `query` (produto escalar) decrescente, descarta
 * qualquer um abaixo de `minScore`, e retorna no máximo `limit`. Pura — mesma entrada, mesma
 * saída; quem chama decide o que é `T` (ex.: o texto do fato).
 */
export function selectTopMatches<T>(
  candidates: readonly Candidate<T>[],
  query: Float32Array,
  opts: { limit: number; minScore: number },
): ScoredMatch<T>[] {
  return candidates
    .map(({ item, embedding }) => ({ item, score: dotProduct(query, embedding) }))
    .sort((a, b) => b.score - a.score)
    .filter((match) => match.score >= opts.minScore)
    .slice(0, opts.limit);
}

/**
 * Serializa um vetor de embedding para gravação em coluna `BLOB` (research.md item 3).
 * `node:sqlite` aceita `Uint8Array` como parâmetro e devolve `Uint8Array` na leitura — `Buffer`
 * (usado aqui só para construir os bytes) é uma subclasse compatível.
 */
export function floatArrayToBuffer(vector: Float32Array): Uint8Array {
  return Buffer.from(vector.buffer, vector.byteOffset, vector.byteLength);
}

/** Caminho inverso de `floatArrayToBuffer` — reconstrói o vetor a partir do `BLOB` lido (`Uint8Array`). */
export function bufferToFloatArray(bytes: Uint8Array): Float32Array {
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / Float32Array.BYTES_PER_ELEMENT);
}

/**
 * Prefixa os fatos recuperados antes de `input` — é isso que vira o `input` final passado a
 * `ReasoningStrategy.run(...)`, por cima da composição de histórico já feita por
 * `composePrompt` (`006-conversation-history`). Sem fatos, retorna `input` inalterado. Pura.
 */
export function composeWithFacts(facts: readonly string[], input: string): string {
  if (facts.length === 0) {
    return input;
  }

  const factsText = facts.map((fact) => `- ${fact}`).join("\n");

  return ["Fatos conhecidos sobre quem está falando:", factsText, "", input].join("\n");
}
