const assert=require('node:assert/strict');
const {parseRows,positionKey,rectForMove,responseMatches,shareCode}=require('./barricade-live-coach.user.js');
assert.deepEqual(parseRows('1. e2 e8 2. e3 e7 3. e4 e6 4. e5'),['e2','e8','e3','e7','e4','e6','e5']);
assert.deepEqual(parseRows('1.e2e82.e3e7'),['e2','e8','e3','e7']);
for(const s of ['chat: 1.e2','2.e3e7','1.e22.e3e7','1.he9','1.e2 3.e4'])assert.equal(parseRows(s),null);
const grid={xs:Array.from({length:9},(_,i)=>i*100),ys:Array.from({length:9},(_,i)=>i*100),w:80,h:80,gapX:20,gapY:20};
for(const reverseX of [false,true])for(const reverseY of [false,true]) {
  const g={...grid,reverseX,reverseY};
  const pawn=rectForMove('d2',g);
  assert.equal(pawn.left,reverseX?500:300);assert.equal(pawn.top,reverseY?700:100);
  const h=rectForMove('ha1',g),v=rectForMove('va1',g);
  assert.equal(h.left,reverseX?700:0);assert.equal(h.top,reverseY?780:80);
  assert.equal(v.left,reverseX?780:80);assert.equal(v.top,reverseY?700:0);
}
const p={red:'e5',blue:'e6',walls:[],red_left:10,blue_left:10,h:'e2,e8,e3,e7,e4,e6,e5'};
const d={request_id:'1',position:p,history:p.h.split(','),top:[[0,'e4']],legal:['e4','d6','f6','e7']};
assert(responseMatches(d,p,'1'));
assert(!responseMatches(d,p,'2')); // response from an earlier generation
assert(!responseMatches({...d,top:[[0,'d2']]},p,'1'));
assert(!responseMatches(d,{...p,blue:'e4'},'1'));
assert.notEqual(positionKey(p),positionKey({...p,blue_left:0}));
assert.notEqual(positionKey(p),positionKey({...p,h:p.h+',e4'}));
assert.equal(shareCode('/game/4wpcv0'),'4wpcv0');
assert.equal(shareCode('/analysis?game=4wpcv0&ref=web_share_end'),'4wpcv0');
assert.equal(shareCode('/analysis?ref=x'),'');
console.log('Overlay history, rotation, reserves, and stale-response regressions passed');
