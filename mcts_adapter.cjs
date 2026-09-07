// Local adapter for Kyutae Lee's MIT-licensed Quoridor MCTS (see vendor license).
// Only these two pinned local source files are evaluated; no network or packages.
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
for (const name of ['game.js', 'ai.js']) {
  vm.runInThisContext(fs.readFileSync(path.join(__dirname, 'vendor/quoridor-ai', name), 'utf8'), {filename: name});
}
const letters = 'abcdefghi';
function decode(m) {
  if (!/^(?:[a-i][1-9]|[hv][a-h][1-8])$/.test(m)) throw new Error('Invalid notation');
  if (m.length === 2) return [[9-Number(m[1]), letters.indexOf(m[0])], null, null];
  const p = [8-Number(m[2]), letters.indexOf(m[1])];
  return m[0] === 'h' ? [null,p,null] : [null,null,p];
}
function encode(move) {
  if (move[0]) return letters[move[0][1]] + (9-move[0][0]);
  const p = move[1] || move[2];
  return (move[1] ? 'h' : 'v') + letters[p[1]] + (8-p[0]);
}
function build(history) {
  const game = new Game(true);
  for (const m of history) {
    if (game.winner !== null) throw new Error('Game already finished');
    const move = decode(m);
    const slot = move[1] || move[2];
    if (slot && (game.pawnOfTurn.numberOfLeftWalls <= 0 ||
        !game.validNextWalls[move[1] ? 'horizontal' : 'vertical'][slot[0]][slot[1]])) {
      throw new Error('Illegal wall: ' + m);
    }
    if (!game.doMove(move, true)) throw new Error('Illegal move: ' + m);
  }
  return game;
}
function inspect(history) {
  const game = build(history);
  const legal = game.winner === null ? game.getArrOfValidNextPositionTuples().map(p=>encode([p,null,null])) : [];
  if (game.winner === null && game.pawnOfTurn.numberOfLeftWalls > 0) {
    legal.push(...game.getArrOfValidNoBlockNextHorizontalWallPositions().map(p=>encode([null,p,null])));
    legal.push(...game.getArrOfValidNoBlockNextVerticalWallPositions().map(p=>encode([null,null,p])));
  }
  return {pawns: game.board.pawns.map(p=>[p.position.col,8-p.position.row]),
    remaining: game.board.pawns.map(p=>p.numberOfLeftWalls),
    winner: game.winner === null ? null : game.winner.index, legal: legal.sort()};
}
function run(input, emit) {
  if (input.inspect) { emit(input.histories.map(inspect)); return; }
  const game = build(input.history);
  if (game.winner !== null) { emit({candidates:[], simulations:0}); return; }
  if (!Number.isInteger(input.rollouts) || input.rollouts < 2 || input.rollouts > 200000 ||
      !Number.isFinite(input.seconds) || input.seconds <= 0 || input.seconds > 60) throw new Error('Invalid budget');
  // Fixed seed is optional and intended for reproducible comparisons only.
  if (Number.isInteger(input.seed)) {
    let seed = input.seed >>> 0;
    Math.random = () => { seed = (Math.imul(1664525,seed)+1013904223) >>> 0; return seed/4294967296; };
  }
  console.log = () => {}; // Upstream diagnostics must not corrupt JSON output.
  const tree = new MonteCarloTreeSearch(game, 0.2);
  const started = Date.now();
  // A promoted policy may guide root exploration. It never masks legal moves
  // and never supplies outcome counts. Deeper nodes retain upstream UCT.
  if (input.root_priors && typeof input.root_priors === 'object') {
    tree.search(2); // visit then expand the root
    let total = 0;
    for (const child of tree.root.children) total += Math.max(0, Number(input.root_priors[encode(child.move)]) || 0);
    if (total > 0) {
      for (const child of tree.root.children) child.policyPrior = Math.max(0, Number(input.root_priors[encode(child.move)]) || 0) / total;
      const original = Object.getOwnPropertyDescriptor(MNode.prototype, 'uct').get;
      const cpuct = Math.max(0.1, Math.min(5, Number(input.cpuct) || 1.25));
      Object.defineProperty(MNode.prototype, 'uct', {configurable:true, get:function() {
        if (this.parent && this.parent.parent === null && this.policyPrior !== undefined) {
          const q = this.numSims ? this.numWins / this.numSims : 0;
          return q + cpuct * this.policyPrior * Math.sqrt(Math.max(1, this.parent.numSims)) / (1 + this.numSims);
        }
        return original.call(this);
      }});
    }
  }
  while (tree.totalNumOfSimulations < input.rollouts && Date.now()-started < input.seconds*1000) {
    tree.search(Math.min(128, input.rollouts-tree.totalNumOfSimulations));
    const candidates = tree.root.children.filter(n=>n.numSims>0).map(n=>({
      move:encode(n.move), visits:n.numSims, rollout_win_rate:n.winRate,
      policy_prior:n.policyPrior
    })).sort((a,b)=>b.visits-a.visits || a.move.localeCompare(b.move));
    // Preserve a completed batch if the parent must terminate a long rollout.
    emit({candidates, simulations:tree.totalNumOfSimulations, elapsed:(Date.now()-started)/1000});
  }
}
if (require.main === module) {
  try { run(JSON.parse(fs.readFileSync(0,'utf8')), x=>process.stdout.write(JSON.stringify(x)+'\n')); }
  catch(e) { process.stderr.write(String(e)+'\n'); process.exitCode=1; }
}
module.exports = {decode, encode, build, inspect, run};
