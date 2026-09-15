import assert from "node:assert/strict";
import { test } from "node:test";

import { dotProduct } from "../domain/memory.ts";
import { embed } from "./embeddings.ts";

// Usa o modelo real (onnx-community/all-MiniLM-L6-v2-ONNX) — mais lento que o resto da suíte, e
// a primeira execução em uma máquina nova baixa o checkpoint ONNX (research.md item 4/9).

test("embed retorna um Float32Array de 384 posições", async () => {
  const vector = await embed("qualquer texto de teste");

  assert.ok(vector instanceof Float32Array);
  assert.equal(vector.length, 384);
});

test("embed retorna um vetor normalizado (norm ~= 1, normalize: true)", async () => {
  const vector = await embed("outro texto de teste");

  const norm = Math.sqrt(dotProduct(vector, vector));

  assert.ok(Math.abs(norm - 1) < 1e-3, `norma esperada ~1, obtida ${norm}`);
});
