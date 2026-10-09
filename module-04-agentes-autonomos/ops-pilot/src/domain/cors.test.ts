import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { corsHeadersFor, DEFAULT_CORS_ORIGINS, parseAllowedOrigins } from "./cors.ts";

describe("parseAllowedOrigins", () => {
  test("sem configuração usa a origem de desenvolvimento da war room (FR-028)", () => {
    assert.deepEqual(parseAllowedOrigins(undefined), ["http://localhost:5173"]);
    assert.deepEqual(parseAllowedOrigins(""), DEFAULT_CORS_ORIGINS);
    assert.deepEqual(parseAllowedOrigins(" , "), DEFAULT_CORS_ORIGINS);
  });

  test("separa por vírgula, tira espaços e itens vazios", () => {
    assert.deepEqual(parseAllowedOrigins(" https://a.com , http://localhost:5173 ,"), [
      "https://a.com",
      "http://localhost:5173",
    ]);
  });
});

describe("corsHeadersFor", () => {
  const allowlist = ["https://a.com"];

  test("origem permitida recebe allow-origin, vary e expõe X-Request-Id (FR-027)", () => {
    assert.deepEqual(corsHeadersFor("https://a.com", allowlist, { preflight: false }), {
      "Access-Control-Allow-Origin": "https://a.com",
      Vary: "Origin",
      "Access-Control-Expose-Headers": "X-Request-Id",
    });
  });

  test("preflight acrescenta métodos, cabeçalhos e max-age", () => {
    const headers = corsHeadersFor("https://a.com", allowlist, { preflight: true });
    assert.equal(headers["Access-Control-Allow-Origin"], "https://a.com");
    assert.equal(headers["Access-Control-Allow-Methods"], "GET, POST, OPTIONS");
    assert.equal(headers["Access-Control-Allow-Headers"], "Content-Type");
    assert.equal(headers["Access-Control-Max-Age"], "600");
  });

  test("origem fora da lista ou ausente não recebe cabeçalho algum (FR-026)", () => {
    assert.deepEqual(corsHeadersFor("https://malicioso.example", allowlist, { preflight: true }), {});
    assert.deepEqual(corsHeadersFor("https://a.com.malicioso.example", allowlist, { preflight: false }), {});
    assert.deepEqual(corsHeadersFor(undefined, allowlist, { preflight: false }), {});
  });

  test("nunca libera credenciais", () => {
    const headers = corsHeadersFor("https://a.com", allowlist, { preflight: true });
    assert.equal("Access-Control-Allow-Credentials" in headers, false);
  });
});
