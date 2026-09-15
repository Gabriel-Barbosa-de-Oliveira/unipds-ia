import assert from "node:assert/strict";
import { test } from "node:test";

import { bufferToFloatArray, composeWithFacts, dotProduct, floatArrayToBuffer, selectTopMatches } from "./memory.ts";

test("dotProduct de vetores ortogonais é 0", () => {
  const a = new Float32Array([1, 0, 0]);
  const b = new Float32Array([0, 1, 0]);

  assert.equal(dotProduct(a, b), 0);
});

test("dotProduct de um vetor normalizado com ele mesmo é 1", () => {
  const v = new Float32Array([1, 0, 0]);

  assert.equal(dotProduct(v, v), 1);
});

test("selectTopMatches ordena por score decrescente e respeita o limit", () => {
  const query = new Float32Array([1, 0]);
  const candidates = [
    { item: "baixo", embedding: new Float32Array([0.5, 0.866]) }, // score 0.5
    { item: "alto", embedding: new Float32Array([1, 0]) }, // score 1
    { item: "medio", embedding: new Float32Array([0.8, 0.6]) }, // score 0.8
  ];

  const result = selectTopMatches(candidates, query, { limit: 2, minScore: 0 });

  assert.equal(result.length, 2);
  assert.equal(result[0]?.item, "alto");
  assert.equal(result[1]?.item, "medio");
});

test("selectTopMatches descarta qualquer candidato abaixo de minScore", () => {
  const query = new Float32Array([1, 0]);
  const candidates = [
    { item: "relevante", embedding: new Float32Array([1, 0]) }, // score 1
    { item: "irrelevante", embedding: new Float32Array([0, 1]) }, // score 0
  ];

  const result = selectTopMatches(candidates, query, { limit: 3, minScore: 0.3 });

  assert.equal(result.length, 1);
  assert.equal(result[0]?.item, "relevante");
});

test("selectTopMatches retorna lista vazia quando não há candidatos", () => {
  const result = selectTopMatches([], new Float32Array([1, 0]), { limit: 3, minScore: 0.3 });

  assert.deepEqual(result, []);
});

test("floatArrayToBuffer/bufferToFloatArray fazem round-trip exato", () => {
  const original = new Float32Array([0.1, -0.2, 0.3, 384]);

  const restored = bufferToFloatArray(floatArrayToBuffer(original));

  assert.deepEqual(Array.from(restored), Array.from(original));
});

test("composeWithFacts sem fatos retorna o input inalterado", () => {
  const result = composeWithFacts([], "qual é o meu nome?");

  assert.equal(result, "qual é o meu nome?");
});

test("composeWithFacts com fatos inclui todos antes do input", () => {
  const facts = ["Gabriel cuida de pagamentos", "Ana cuida de autenticação"];

  const result = composeWithFacts(facts, "quem cuida do checkout?");

  const indexFact1 = result.indexOf("Gabriel cuida de pagamentos");
  const indexFact2 = result.indexOf("Ana cuida de autenticação");
  const indexInput = result.indexOf("quem cuida do checkout?");

  assert.ok(indexFact1 >= 0 && indexFact2 >= 0 && indexInput >= 0);
  assert.ok(indexFact1 < indexInput && indexFact2 < indexInput);
});
