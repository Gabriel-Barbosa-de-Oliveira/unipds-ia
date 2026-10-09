import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { describe, test } from "node:test";

import { BASE_PATH, localAssetPaths, resolveBasePath } from "./base-path.ts";

const DIST_INDEX = new URL("../../dist/index.html", import.meta.url);

describe("localAssetPaths", () => {
  test("extrai src/href locais e ignora URLs absolutas", () => {
    const html =
      '<link href="/opspilot/assets/a.css"><script src="/opspilot/assets/a.js"></script>' +
      '<link href="https://fonts.example/x.css"><a href="#topo">';
    assert.deepEqual(localAssetPaths(html), ["/opspilot/assets/a.css", "/opspilot/assets/a.js"]);
  });
});

describe("resolveBasePath (spec 016)", () => {
  test("sem valor usa o padrão local", () => {
    assert.equal(resolveBasePath(undefined), BASE_PATH);
    assert.equal(resolveBasePath(""), "/opspilot/");
    assert.equal(resolveBasePath("   "), "/opspilot/");
  });

  test("caminho do Pages passa como está", () => {
    assert.equal(resolveBasePath("/unipds-ia/opspilot/"), "/unipds-ia/opspilot/");
  });

  test("normaliza barras no início, no fim e repetidas", () => {
    assert.equal(resolveBasePath("unipds-ia/opspilot"), "/unipds-ia/opspilot/");
    assert.equal(resolveBasePath("//x//"), "/x/");
    assert.equal(resolveBasePath(" /a//b "), "/a/b/");
  });

  test("rejeita o que não é caminho, citando OPSPILOT_WEB_BASE", () => {
    for (const raw of ["http://x/opspilot/", "/a/../b/", "/com espaço/", "/x?q/", "/x#y/"]) {
      assert.throws(() => resolveBasePath(raw), /OPSPILOT_WEB_BASE/, raw);
    }
  });
});

describe("build publicado sob o caminho base (US5 da 015, SC-006 da 016)", () => {
  const base = resolveBasePath(process.env.OPSPILOT_WEB_BASE);

  test(`todo asset do index.html gerado começa com ${base}`, (t) => {
    if (!existsSync(DIST_INDEX)) {
      t.skip("rode `npm --prefix web run build` antes para validar o build");
      return;
    }
    const paths = localAssetPaths(readFileSync(DIST_INDEX, "utf8"));
    assert.ok(paths.length > 0);
    for (const path of paths) {
      assert.ok(path.startsWith(base), `${path} fora de ${base}`);
    }
  });

  test("vite.config.ts resolve o base por resolveBasePath(OPSPILOT_WEB_BASE)", () => {
    const config = readFileSync(new URL("../../vite.config.ts", import.meta.url), "utf8");
    assert.match(config, /import \{ resolveBasePath \} from "\.\/src\/lib\/base-path\.ts"/);
    assert.match(config, /base: resolveBasePath\(process\.env\.OPSPILOT_WEB_BASE\)/);
  });
});
