// node demo/cli.js model.json windows.json -> prints JSON list of predictions (m/s)
const fs = require("fs"); const M = require("./model.js");
const m = JSON.parse(fs.readFileSync(process.argv[2])); const W = JSON.parse(fs.readFileSync(process.argv[3]));
console.log(JSON.stringify(W.map(w => M.predict(m, w).y)));
