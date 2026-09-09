// ==UserScript==
// @name         Barricade Live Coach
// @namespace    welldonestreams
// @version      1.5.0
// @description  Verified board advice, visible highlights and factual explanations
// @match        https://barricade.gg/*
// @grant        GM_getValue
// @grant        GM_setValue
// @grant        GM_xmlhttpRequest
// @connect      127.0.0.1
// @connect      api.barricade.gg
// @run-at       document-idle
// ==/UserScript==

(() => {
  'use strict';
  const FILES = 'abcdefghi';
  const MOVE = /^(?:[a-i][1-9]|[hv][a-h][1-8])$/;
  // Pure functions are exported only in Node, for independent regression tests.
  function parseRows(text) {
    const compact = text.replace(/\s/g, '');
    const rows = [...compact.matchAll(/(\d+)\.((?:[hv]?[a-i][1-9]){1,2})/g)];
    if (!rows.length || rows.map(r => r[0]).join('') !== compact) return null;
    if (rows.some((r,i) => +r[1] !== i+1)) return null;
    const history = rows.flatMap(r => r[2].match(/[hv]?[a-i][1-9]/g));
    if (history.some(m => !MOVE.test(m))) return null;
    if (rows.slice(0,-1).some(r => r[2].match(/[hv]?[a-i][1-9]/g).length !== 2)) return null;
    return history;
  }
  function positionKey(p) {
    return JSON.stringify([p.red,p.blue,[...p.walls].sort(),p.red_left,p.blue_left,p.h]);
  }
  function rectForMove(move, grid) {
    if (!MOVE.test(move)) return null;
    const wall = move.length===3, offset=wall?1:0;
    const file=FILES.indexOf(move[offset]), rank=+move[offset+1];
    const col=grid.reverseX?8-file:file, row=grid.reverseY?9-rank:rank-1;
    if (!wall) return {left:grid.xs[col],top:grid.ys[row],width:grid.w,height:grid.h};
    const leftCol=Math.min(col,grid.reverseX?col-1:col+1);
    const topRow=Math.min(row,grid.reverseY?row-1:row+1);
    return move[0]==='h' ?
      {left:grid.xs[leftCol],top:grid.ys[topRow]+grid.h,width:2*grid.w+grid.gapX,height:grid.gapY} :
      {left:grid.xs[leftCol]+grid.w,top:grid.ys[topRow],width:grid.gapX,height:2*grid.h+grid.gapY};
  }
  function responseMatches(data, snapshot, id) {
    return data.request_id===id && data.history.join(',')===snapshot.h &&
      ['red','blue','red_left','blue_left'].every(k => data.position[k]===snapshot[k]) &&
      JSON.stringify(data.position.walls)===JSON.stringify(snapshot.walls) &&
      data.top.every(t=>data.legal.includes(t[1]));
  }
  if (typeof module!=='undefined' && module.exports) {
    module.exports={parseRows,positionKey,rectForMove,responseMatches}; return;
  }
  let COACH=null;
  const mine=n=>/^steak/i.test(n)||[GM_getValue('bc_me','steak2222'),...GM_getValue('bc_alts','').split(',')].some(m=>m.toLowerCase()===n.toLowerCase());
  let myColor=null, opponent='', gameId='', manual=false, enabled=true;
  let current=null, stableKey='', stableTicks=0, generation=0, request=null, lastAdvice=null;
  let analyzedFor=null;
  let highlights=[], retryAt=0;
  const fetchedOpponents=new Set();
  const cache=new Map();
  const panel=document.createElement('div'); panel.id='bc-coach';
  panel.style.cssText='position:fixed;right:16px;top:16px;z-index:2147483000;background:#111827f5;color:#f9fafb;border:1px solid #64748b;border-radius:12px;padding:12px;max-width:350px;font:14px system-ui;box-shadow:0 4px 20px #0008';
  panel.innerHTML='<div style="display:flex;gap:6px;align-items:center"><strong>Coach 1.5</strong><button id="bc-red">Red</button><button id="bc-blue">Blue</button><button id="bc-on">Pause</button></div><div id="bc-move" style="font-size:21px;color:#6ee7b7;margin-top:8px">Reading board…</div><div id="bc-status" style="font-size:12px;margin:6px 0"></div><div id="bc-why" style="font-size:13px;line-height:1.5"></div><div id="bc-opp" style="font-size:12px;color:#cbd5e1;margin-top:8px"></div>';
  document.body.appendChild(panel);
  const el=id=>panel.querySelector('#bc-'+id);
  function clear() { highlights.forEach(e=>e.remove()); highlights=[]; }
  function invalidate(message) {
    generation++; if(request) request.abort(); request=null;
    clear(); lastAdvice=null; el('move').textContent='—'; el('why').textContent=''; el('status').textContent=message; el('opp').textContent=opponent?'vs '+opponent:'';
  }
  function setColor(color, isManual=false) {
    if(isManual) {manual=true;GM_setValue('bc_color_'+gameId,color);}
    if(color!==myColor) {myColor=color;analyzedFor=null;invalidate('Color updated');stableTicks=0;}
    for(const c of ['red','blue']) {el(c).setAttribute('aria-pressed',String(c===myColor));el(c).style.background=c===myColor?'#047857':'#374151';el(c).style.color='white';}
  }
  el('red').onclick=()=>setColor('red',true); el('blue').onclick=()=>setColor('blue',true);
  el('on').onclick=()=>{enabled=!enabled;el('on').textContent=enabled?'Pause':'Resume';invalidate(enabled?'Reading board…':'Paused');stableTicks=0;};
  function nearest(values,n) {return values.reduce((best,v,i)=>Math.abs(v-n)<Math.abs(values[best]-n)?i:best,0);}
  function visible(node) {const r=node.getBoundingClientRect(),s=getComputedStyle(node);return r.width>0&&r.height>0&&s.visibility!=='hidden'&&+s.opacity>.9;}
  function readBoard() {
    // Locate an actual pawn by size, not the red badge in a player card.
    for(const pawn of document.querySelectorAll('[class*="bg-red-500"]')) {
      const pr=pawn.getBoundingClientRect(); if(pr.width<15||Math.abs(pr.width-pr.height)>3) continue;
      let root=pawn.parentElement;
      for(let depth=0;root&&depth<7;depth++,root=root.parentElement) {
        const cells=[...root.querySelectorAll('div')].filter(n=>{
          const r=n.getBoundingClientRect(),cl=String(n.className);
          return cl.includes('border')&&cl.includes('flex')&&r.width>30&&r.width<220&&Math.abs(r.width-r.height)<3;
        });
        if(cells.length!==81) continue;
        const xs=[...new Set(cells.map(c=>Math.round(c.getBoundingClientRect().left)))].sort((a,b)=>a-b);
        const ys=[...new Set(cells.map(c=>Math.round(c.getBoundingClientRect().top)))].sort((a,b)=>a-b);
        if(xs.length!==9||ys.length!==9) continue;
        const r=cells[0].getBoundingClientRect(),w=r.width,h=r.height;
        const cx=xs.map(x=>x+w/2),cy=ys.map(y=>y+h/2);
        // CSS grid cell 2/2 is a9 in the site's canonical board. Its physical
        // location reveals rotation independently of your color.
        const canonical=cells.find(n=>n.style.gridRowStart==='2'&&n.style.gridColumnStart==='2');
        if(!canonical||cells.some(n=>!/^([2468]|1[02468])$/.test(n.style.gridRowStart)||!/^([2468]|1[02468])$/.test(n.style.gridColumnStart))) continue;
        const ar=canonical.getBoundingClientRect();
        const reverseX=Math.abs(ar.left-xs[8])<3,reverseY=Math.abs(ar.top-ys[0])<3;
        const fileOrder=reverseX?'ihgfedcba':'abcdefghi',rankOrder=reverseY?'987654321':'123456789';
        const grid={xs,ys,w,h,gapX:xs[1]-xs[0]-w,gapY:ys[1]-ys[0]-h,reverseX,reverseY};
        const square=node=>{const r=node.getBoundingClientRect();return fileOrder[nearest(cx,r.left+r.width/2)]+rankOrder[nearest(cy,r.top+r.height/2)];};
        const blue=[...root.querySelectorAll('[class*="bg-blue-500"]')].find(n=>{const r=n.getBoundingClientRect();return r.width>15&&Math.abs(r.width-r.height)<3;});
        if(!blue) continue;
        const walls=new Set();
        // Placed bars have stable test IDs; hover previews have no wall-bar ID.
        for(const n of root.querySelectorAll('[data-testid^="wall-bar-"]')) {
          const m=n.getAttribute('data-testid').match(/^wall-bar-(horizontal|vertical)-([0-7])-([0-7])$/);
          if(m) walls.add((m[1]==='horizontal'?'h':'v')+FILES[+m[3]]+(8-(+m[2])));
        }
        return {red:square(pawn),blue:square(blue),walls:[...walls].sort(),grid,root};
      }
    }
    return null;
  }
  function readHistory(board) {
    let best=null;
    for(const node of document.querySelectorAll('div,ol,table')) {
      if(panel.contains(node)||board.root.contains(node)||!visible(node)) continue;
      // textContent also handles the move-list's concatenated inline spans.
      const t=node.textContent.trim(); if(t.length>8000||!/^1\s*\./.test(t)) continue;
      const parsed=parseRows(t);
      if(parsed&&(!best||parsed.length>best.length)) best=parsed;
    }
    if(!best&&board.red==='e1'&&board.blue==='e9'&&!board.walls.length) return [];
    return best;
  }
  function readCards(board) {
    const cards={};
    for(const color of ['red','blue']) {
      for(const badge of document.querySelectorAll(`[class*="bg-${color}-500"]`)) {
        if(board.root.contains(badge)||panel.contains(badge)) continue;
        let node=badge;
        for(let i=0;node&&i<6;i++,node=node.parentElement) {
          const text=node.textContent;
          const counts=[...text.matchAll(/(\d{1,2})\s*\/\s*10/g)];
          if(counts.length!==1) continue;
          const links=[...node.querySelectorAll('a[href*="/user/"]')];
          let name=links.length===1?decodeURIComponent(links[0].getAttribute('href').split('/user/')[1].split(/[?#]/)[0]):null;
          if(!name) {const m=text.match(/([A-Za-z0-9_-]+)\s*\(\s*\d{3,4}\s*\)/);if(m)name=m[1];}
          if(name) {cards[color]={name,left:+counts[0][1]};break;}
        }
        if(cards[color]) break;
      }
    }
    if(!cards.red||!cards.blue) return null;
    if(!manual) {const color=['red','blue'].find(c=>mine(cards[c].name));if(color)setColor(color);}
    if(myColor) {
      opponent=cards[myColor==='red'?'blue':'red'].name;
      if(!fetchedOpponents.has(opponent)) {
        fetchedOpponents.add(opponent);
        GM_xmlhttpRequest({method:'GET',url:COACH+'/api/opponent/fetch?name='+encodeURIComponent(opponent),timeout:3000});
      }
    }
    return cards;
  }
  function draw(move,grid) {
    clear(); const r=rectForMove(move,grid);if(!r)return;
    const box=document.createElement('div');
    box.style.cssText=`position:fixed;pointer-events:none;z-index:2147482999;left:${r.left}px;top:${r.top}px;width:${r.width}px;height:${r.height}px;box-sizing:border-box;border:4px solid #34d399;border-radius:6px;background:#34d39933;box-shadow:0 0 16px #34d399`;
    const label=document.createElement('div'); label.textContent='Coach: '+move;
    label.style.cssText=`position:fixed;pointer-events:none;z-index:2147482999;left:${r.left}px;top:${r.top-25}px;background:#34d399;color:#052e16;padding:3px 6px;font:bold 13px system-ui;border-radius:4px`;
    document.body.append(box,label);highlights=[box,label];
  }
  function recordGame(data, snapshot) {
    if(!snapshot || !snapshot.h) return;
    const me = GM_getValue('bc_me','steak2222');
    const oppName = opponent || 'unknown';
    const winnerName = data.winner === 0 ? 'red' : 'blue';
    const red = snapshot.red_name || (myColor === 'red' ? me : oppName);
    const blue = snapshot.blue_name || (myColor === 'blue' ? me : oppName);
    const p = new URLSearchParams({h: snapshot.h, winner: winnerName, red, blue});
    GM_xmlhttpRequest({method:'GET', url: COACH + '/api/record?' + p, timeout: 4000,
      onload(){}, onerror(){}, ontimeout(){}});
  }
  function analyzeGame(data, snapshot) {
    // Idempotent: the same finished game can be re-detected every tick (both the
    // tick-level goal check and present()'s winner branch), and the server's live
    // search can briefly hold the lock. Fire once per game; retry on 429.
    if(analyzedFor === gameId) return;
    analyzedFor = gameId;
    const firedFor = gameId, reviewColor = myColor;
    if(!myColor) { invalidate('Game finished'); return; }
    if(!snapshot || !snapshot.h) { invalidate('Game finished'); return; }
    invalidate('Game finished — verifying review');
    const winnerName = data.winner === 0 ? 'red' : 'blue';
    el('move').textContent = winnerName === myColor ? 'You won' : 'You lost';
    el('status').textContent = 'Analyzing this game…';
    el('why').textContent = '';
    const p = new URLSearchParams({h:snapshot.h, side:myColor, red:snapshot.red, blue:snapshot.blue,
      walls:snapshot.walls.join(','), red_left:snapshot.red_left, blue_left:snapshot.blue_left, depth:'2',seconds:'0.6'});
    const attempt = (tries) => {
      if(gameId!==firedFor||myColor!==reviewColor||!enabled)return;
      GM_xmlhttpRequest({method:'GET', url: COACH + '/api/analyze?' + p, timeout: 18000,
        onload(res) {
          if(gameId !== firedFor || myColor!==reviewColor || !enabled) return;   // navigated away; don't touch the pill
          try {
            const d = JSON.parse(res.responseText);
            if (res.status === 429) {
              if (tries > 0) { el('status').textContent = 'Coach busy — retrying analysis…'; setTimeout(() => attempt(tries-1), 1500); return; }
              throw Error(d.error || 'coach busy');
            }
            if (res.status !== 200 || d.error) throw Error(d.error || 'analysis failed');
            if (d.winner !== data.winner) throw Error('Game result and history disagree');
            recordGame(data,snapshot);
            if (!d.blunders || !d.blunders.length) {
              el('status').textContent = (d.complete?'Review complete':'Partial review')+' — no errors established at depth 2 in the reviewed moves.';
              return;
            }
            const lines = d.blunders.map(b =>
              `ply ${b.ply}: you played ${b.move} — coach had ${b.best} (Δ${b.delta})`).join('\n');
            el('status').textContent = `${d.complete?'Review':'Partial review'} candidates (${d.side}):`;
            el('why').textContent = lines;
          } catch(e) {
            el('status').textContent = 'Analysis unavailable.';
            el('why').textContent = e.message;
          }
        },
        onerror(){ if(gameId===firedFor) el('status').textContent = 'Analysis unavailable.'; },
        ontimeout(){ if(gameId===firedFor) el('status').textContent = 'Analysis timed out.'; }
      });
    };
    attempt(3);
  }
  function present(data,snapshot) {
    if(data.winner!==null) {analyzeGame(data,snapshot);return;}
    if(!myColor) {invalidate('Choose your color');return;}
    if(data.to_move!==myColor) {invalidate('Opponent to move');return;}
    if(data.search?.forced_loss || data.search?.outcome==='opponent forced goal') {
      el('move').textContent='LOST — no saving move';
      el('move').style.color='#fca5a5';
      el('why').textContent=data.why || 'Every legal move loses with best play.';
      clear();
      return;
    }
    const best=data.top[0]?.[1];if(!best)return;
    lastAdvice={data,key:positionKey(snapshot)};
    el('move').textContent=(data.search?.position_warning?'Best try: ':'Play ')+best;
    el('move').style.color=data.search?.position_warning?'#fbbf24':'#6ee7b7';
    el('status').textContent=`You are ${myColor} · ${(data.search.elapsed).toFixed(2)}s`;
    el('why').textContent=data.why; draw(best,snapshot.grid);
    const used=data.opponent_evidence.some(r=>r.move===best);
    el('opp').textContent=opponent?`vs ${opponent} · ${used?'matching past replies used to break a tactical tie':'no matching reply evidence used'}`:'';
  }
  function ask(snapshot) {
    const key=gameId+'|'+opponent+'|'+positionKey(snapshot);
    if(cache.has(key)) {present(cache.get(key),snapshot);return;}
    const id=String(++generation);
    const params=new URLSearchParams({...snapshot,grid:'',root:'',opponent,game_id:gameId,seconds:'4',request_id:id,walls:snapshot.walls.join(',')});
    el('status').textContent='Thinking…';
    request=GM_xmlhttpRequest({method:'GET',url:COACH+'/api/live?'+params,timeout:18000,
      onload(res) {
        if(String(generation)!==id)return;request=null;
        try {
          const data=JSON.parse(res.responseText);
          if(res.status!==200||data.error)throw Error(data.error||'Coach request failed');
          const freshBoard=readBoard(),freshHist=freshBoard&&readHistory(freshBoard),freshCards=freshBoard&&readCards(freshBoard);
          const fresh=freshBoard&&freshHist&&freshCards?{...freshBoard,h:freshHist.join(','),red_left:freshCards.red.left,blue_left:freshCards.blue.left}:null;
          if(!fresh||positionKey(fresh)!==positionKey(snapshot)||!current||positionKey(current)!==positionKey(snapshot)||!responseMatches(data,snapshot,id))throw Error('Position changed; discarding old advice');
          cache.set(key,data);if(cache.size>100)cache.delete(cache.keys().next().value);
          present(data,current);
        } catch(e) {invalidate(e.message);retryAt=Date.now()+1200;}
      },
      onerror(){if(String(generation)===id){request=null;invalidate('Start the updated coach server');retryAt=Date.now()+3000;}},
      ontimeout(){if(String(generation)===id){request=null;invalidate('18-second limit reached; no stale advice shown');retryAt=Date.now()+1000;}}
    });
  }
  function tick() {
    if(!enabled||!COACH)return;
    const url=location.pathname+location.search;
    if(url!==gameId){gameId=url;manual=false;myColor=null;opponent='';cache.clear();current=null;stableKey='';stableTicks=0;analyzedFor=null;invalidate('Reading this game');const saved=GM_getValue('bc_color_'+gameId,null);if(saved==='red'||saved==='blue')setColor(saved,true);}
    const board=readBoard();
    if(!board){if(current)invalidate('Waiting for a readable board');current=null;return;}
    const hist=readHistory(board),cards=readCards(board);
    if(!hist||!cards){if(current)invalidate('Waiting for the numbered move list and wall counts');current=null;clear();return;}
    const snapshot={...board,h:hist.join(','),red_left:cards.red.left,blue_left:cards.blue.left,red_name:cards.red.name,blue_name:cards.blue.name};
    // Game over: a pawn on its goal rank. Detect it directly from the board so
    // the post-game review fires no matter whose turn it was or whether the
    // move list was fully readable.
    const redOnGoal = /9$/.test(board.red), blueOnGoal = /1$/.test(board.blue);
    if(redOnGoal||blueOnGoal) {
      // Game finished: freeze this game's tick loop so we stop polling /api/live
      // (which would hold the search lock and starve the analysis), and only
      // kick off the review once.
      analyzeGame({winner: redOnGoal ? 0 : 1}, snapshot);
      return;
    }
    const key=positionKey(snapshot);current=snapshot;
    if(key!==stableKey){stableKey=key;stableTicks=0;invalidate('Checking board…');return;}
    stableTicks++;
    if(lastAdvice&&lastAdvice.key===key){draw(lastAdvice.data.top[0][1],board.grid);return;}
    if(stableTicks<2||request||Date.now()<retryAt||!myColor)return;
    const side=hist.length%2?'blue':'red';
    if(side!==myColor){el('status').textContent='Opponent to move';return;}
    ask(snapshot);
  }
  // No guessed transitions, predictive history, or “you played” accusations.
  // Only connect to a server with these safeguards. An elevated older server
  // may own 8810; the updated local fallback can safely use 8811.
  (async()=>{
    for(const port of [8810,8811]) {
      const base='http://127.0.0.1:'+port;
      const compatible=await new Promise(resolve=>GM_xmlhttpRequest({method:'GET',url:base+'/api/health',timeout:1500,
        onload:r=>{try{const h=JSON.parse(r.responseText);resolve(h.service==='barricade-coach'&&h.live_protocol>=7)}catch{resolve(false)}},
        onerror:()=>resolve(false),ontimeout:()=>resolve(false)}));
      if(compatible){COACH=base;setInterval(tick,300);return;}
    }
    el('status').textContent='Run START-COACH.cmd to start the updated coach, then reload this page.';
  })();
})();
