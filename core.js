// Arabic normalisation + word alignment. Shared by browser and node tests.
export function normalize(s){
  return s
    .replace(/َى(?=[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]*(\s|$))/g,'َا').replace(/ىٰ/g,'ا').replace(/ٰ/g,'ا')
    .replace(/[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u08D3-\u08FF\u0640]/g,'')
    .replace(/[ٱأإآ]/g,'ا').replace(/ى/g,'ي').replace(/ة/g,'ه')
    .replace(/ؤ/g,'و').replace(/ئ/g,'ي').replace(/[ءا]/g,'').replace(/[^\u0621-\u064A\s]/g,' ')
    .replace(/\s+/g,' ').trim();
}
function lev(a,b){
  const m=a.length,n=b.length; if(!m) return n; if(!n) return m;
  let p=Array.from({length:n+1},(_,i)=>i);
  for(let i=1;i<=m;i++){const c=[i];for(let j=1;j<=n;j++)c[j]=Math.min(p[j]+1,c[j-1]+1,p[j-1]+(a[i-1]===b[j-1]?0:1));p=c;}
  return p[n];
}
// similarity cost 0..1 between expected word and heard word
function cost(a,b){ if(a===b) return 0; const d=lev(a,b); return Math.min(1,d/Math.max(a.length,b.length)); }
// expected: array of raw words (with tashkeel). heard: transcript string.
// returns {words:[{text,status,heard}], extras:[...], score}
export function align(expected, heardText){
  const raw=heardText.split(/\s+/).filter(t=>normalize(t)); const E=expected.map(normalize), H=raw.map(normalize);
  const m=E.length,n=H.length, GAP=0.8;
  const D=Array.from({length:m+1},()=>new Float32Array(n+1)), T=Array.from({length:m+1},()=>new Uint8Array(n+1));
  for(let i=1;i<=m;i++){D[i][0]=i*GAP;T[i][0]=1;}
  for(let j=1;j<=n;j++){D[0][j]=j*GAP;T[0][j]=2;}
  for(let i=1;i<=m;i++)for(let j=1;j<=n;j++){
    const c=cost(E[i-1],H[j-1]);
    const sub=D[i-1][j-1]+c*1.0, del=D[i-1][j]+GAP, ins=D[i][j-1]+GAP;
    if(sub<=del&&sub<=ins){D[i][j]=sub;T[i][j]=0;}else if(del<=ins){D[i][j]=del;T[i][j]=1;}else{D[i][j]=ins;T[i][j]=2;}
  }
  const words=expected.map(t=>({text:t,status:'missing',heard:''})), extras=[];
  let i=m,j=n;
  while(i>0||j>0){
    const t=(i>0&&j>0)?T[i][j]:(i>0?1:2);
    if(t===0){const c=cost(E[i-1],H[j-1]);
      words[i-1].heard=raw[j-1];words[i-1].heardN=H[j-1];
      words[i-1].status=c===0?'correct':(c<=0.34?'close':'wrong');
      i--;j--;}
    else if(t===1){i--;}
    else{extras.push({after:i-1,word:raw[j-1]});j--;}
  }
  extras.reverse();
  const ok=words.filter(w=>w.status==='correct').length+0.5*words.filter(w=>w.status==='close').length;
  return {words,extras,score:m?Math.round(100*ok/m):0};
}

// split an ayah into display words; waqf/pause marks attach to the previous word
export function tokens(text){
  const out=[];
  for(const t of text.split(/\s+/).filter(Boolean)){
    if(!normalize(t)&&out.length) out[out.length-1]+=' '+t; else out.push(t);
  }
  return out;
}
// vowel signature of a word: consonants only, alef/hamza skipped; shadda & sukun ignored.
function vowels(w){
  const v=[];let initial='',first=true,cur=null;
  for(const ch of w.replace(/َى(?=[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]*(\s|$))/g,'َا').replace(/ىٰ/g,'ا')){
    const c=ch.codePointAt(0);
    if(c>=0x64B&&c<=0x650){ // tanween/fatha/damma/kasra
      const k=(c===0x64B||c===0x64E)?'a':(c===0x64C||c===0x64F)?'u':'i';
      if(cur==='cons') v[v.length-1]=k; else if(cur==='init') initial=k;
    } else if(/[\u0621-\u064A]/.test(ch)&&!/[ءاأإآٱ]/.test(ch)){v.push('');cur='cons';first=false;}
    else if(/[ءاأإآٱ]/.test(ch)){cur=first?'init':'skip';first=false;}
  }
  return {v:v.join('|'),initial};
}
export function vowelMismatch(expected,heard){
  const a=vowels(expected),b=vowels(heard);
  if(a.v!==b.v){
    // end-of-word vowel often differs by pause (waqf); ignore the last vowel
    const x=a.v.split('|'),y=b.v.split('|');
    if(x.length===y.length&&x.slice(0,-1).join('|')===y.slice(0,-1).join('|')) return false;
    return true;
  }
  return !!(a.initial&&b.initial&&a.initial!==b.initial);
}
