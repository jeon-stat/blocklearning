// tests/test_engine.py가 부름: 로짓 줄마다 엔진의 topProbs(분류 확률 상위 k개)를 계산해 출력 (predict.top_probs와 맞추기).
const fs = require("fs");
const path = require("path");
globalThis.tf = require("@tensorflow/tfjs");
const E = require(path.join(__dirname, "../../src/blocklearning/static/engine.js"));
const f = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
process.stdout.write(JSON.stringify(f.rows.map((r) => E.topProbs(Float32Array.from(r), f.classes, f.k))));
