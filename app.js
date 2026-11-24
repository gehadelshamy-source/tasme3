import {align,tokens,vowelMismatch} from './core.js';
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.wasm.min.mjs';
import {computeFeatures} from './mel.js';
import {forcedScores} from './ctc.js';

const MODEL_URL='https://huggingface.co/mrmuminov/quran-fastconformer-ctc-ar/resolve/main/onnx/model_int8.onnx';
const TH=-3; // forced-alignment threshold (min peak log-prob of a character in a word)
const $=id=>document.getElementById(id);
const status=t=>$('status').textContent=t;
let Q=[], words=[], sess=null, sessP=null, chars=null, fb=null;
let heardParts=[], lpParts=[], queue=Promise.resolve(), live=false, finalized=false;
const FAST=location.search.includes('fast=1');
ort.env.wasm.wasmPaths='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
ort.env.wasm.numThreads=self.crossOriginIsolated?Math.min(4,navigator.hardwareConcurrency||2):1;

async function fetchModel(){
  const cache=await caches.open('tasme3-model');
  let r=await cache.match(MODEL_URL);
  if(r) return new Uint8Array(await r.arrayBuffer());
  const resp=await fetch(MODEL_URL);
  if(!resp.ok) throw new Error('HTTP '+resp.status);
  const total=+resp.headers.get('content-length')||174e6, rd=resp.body.getReader(), parts=[];let got=0;
  for(;;){const {done,value}=await rd.read();if(done)break;parts.push(value);got+=value.length;status('تحميل النموذج '+Math.round(got/total*100)+'% (مرة واحدة فقط، ١٧٤ ميجا)');}
  const bytes=new Uint8Array(got);let o=0;for(const p of parts){bytes.set(p,o);o+=p.length;}
  try{await cache.put(MODEL_URL,new Response(bytes));}catch{}
  return bytes;
}
function loadModel(){
  if(sessP) return sessP;
  status('بنحمّل النموذج…');
  sessP=(async()=>{
    chars=await (await fetch('vocab.json')).json();
    fb=new Float32Array(await (await fetch('https://huggingface.co/mrmuminov/quran-fastconformer-ctc-ar/resolve/main/onnx/fb.bin')).arrayBuffer());
    const bytes=await fetchModel();
    status('بيجهّز النموذج…');
    sess=await ort.InferenceSession.create(bytes,{executionProviders:['wasm']});
    status('جاهز ✔ (بيشتغل على جهازك، الصوت مبيتبعتش لأي مكان)'+(self.crossOriginIsolated?' ⚡':''));
    return sess;
  })().catch(e=>{sessP=null;status('فشل تحميل النموذج: '+e.message);throw e;});
  return sessP;
}

async function init(){
  Q=(await Promise.all(Array.from({length:7},(_,i)=>fetch('q'+i+'.json').then(r=>r.json())))).flat();
  const s=$('surah'); Q.forEach((x,i)=>{const o=document.createElement('option');o.value=i;o.textContent=(i+1)+' - '+x.n;s.appendChild(o);});
  const saved=JSON.parse(localStorage.getItem('sel')||'{}');
  if(saved.s!=null){s.value=saved.s;$('from').value=saved.f;$('to').value=saved.t;$('hidden').checked=!!saved.h;}
  show();
  if('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(()=>{});
  loadModel().catch(()=>{});
}

function show(){
  const si=+$('surah').value, a=Q[si].a;
  let f=Math.max(1,Math.min(+$('from').value||1,a.length)), t=Math.max(f,Math.min(+$('to').value||f,a.length));
  $('from').value=f;$('to').value=t;
  localStorage.setItem('sel',JSON.stringify({s:si,f,t,h:$('hidden').checked}));
  words=[];
  for(let n=f;n<=t;n++) tokens(a[n-1]).forEach(w=>words.push({text:w,ayah:n,status:'pending',heard:''}));
  heardParts=[];lpParts=[];finalized=false;$('summary').textContent='';$('detail').hidden=true;
  render(null);
}

function render(res){
  const box=$('text');box.innerHTML='';let lastAyah=0;
  const hide=$('hidden').checked;
  words.forEach((w,i)=>{
    if(w.ayah!==lastAyah){lastAyah=w.ayah;const n=document.createElement('span');n.className='num';n.textContent=' ﴿'+w.ayah.toLocaleString('ar-EG')+'﴾ ';if(i>0)box.appendChild(n);}
    const s=document.createElement('span');
    s.className='w '+w.status+(hide&&w.status==='pending'?' hide':'')+(hide&&w.status!=='pending'?' hide':'');
    s.textContent=w.text;
    if(w.heard&&(w.status==='wrong'||w.status==='close'||w.status==='tashkeel')){const h=document.createElement('span');h.className='heard';h.textContent=w.heard;s.appendChild(h);}
    s.onclick=()=>detail(w);
    box.appendChild(s);
  });
}
function detail(w){
  const d=$('detail');
  const m={correct:'صح ✔',close:'قريب من الصح',tashkeel:'الحروف صح بس التشكيل (الحركات) مختلف',wrong:'غلط',missing:'اتنسى (ما اتسمعش)',pending:'لسه'}[w.status];
  d.hidden=false;d.innerHTML=`<b>${w.text}</b> : ${m}`+(w.heard?`<br>سمعت: <b>${w.heard}</b>`:'');
  clearTimeout(detail.t);detail.t=setTimeout(()=>d.hidden=true,5000);
}

function apply(final){
  const r=align(words.map(w=>w.text),heardParts.join(' '));
  let last=-1;r.words.forEach((x,i)=>{if(x.status!=='missing')last=i;});
  const n=final?words.length:last+1;
  let sc=[];
  if(n>0&&lpParts.length&&chars){
    const V=chars.length+1,T=lpParts.reduce((q,p)=>q+p.T,0),lp=new Float32Array(T*V);let o=0;
    for(const p of lpParts){lp.set(p.lp,o*V);o+=p.T;}
    sc=forcedScores(lp,T,V,words.slice(0,n).map(w=>w.text),chars);
  }
  words.forEach((w,i)=>{
    const x=r.words[i],s=sc[i];
    if(!final&&i>last){w.status='pending';w.heard='';return;}
    w.status=x.status;w.heard=x.heard;
    if(s){
      if(s.min>=TH){w.status='correct';w.heard='';}
      else if(x.status==='correct'){w.status=vowelMismatch(w.text,x.heard)?'tashkeel':'correct';}
      else if(x.status!=='missing'&&s.minLetter>=TH&&s.minMark<TH){w.status='tashkeel';}
    } else if(x.status==='correct'&&vowelMismatch(w.text,x.heard)) w.status='tashkeel';
  });
  const wrong=words.filter(w=>w.status==='wrong').length,close=words.filter(w=>w.status==='close').length,miss=words.filter(w=>w.status==='missing').length;
  const ok=words.filter(w=>w.status==='correct').length,tk=words.filter(w=>w.status==='tashkeel').length;
  $('summary').textContent=final?`النتيجة: ${r.score}% — صح ${ok} | قريب ${close} | تشكيل ${tk} | غلط ${wrong} | ناقص ${miss} | زيادة ${r.extras.length}`:`بتسمّع… (${ok} كلمة صح لحد دلوقتي)`;
  render(r);
}

async function transcribe(pcm){
  await loadModel();
  let e=0;for(let i=0;i<pcm.length;i+=16)e+=pcm[i]*pcm[i];e/=pcm.length/16;
  if(e<1e-6) return null;
  const t0=performance.now();
  const {features,frames,length}=computeFeatures(pcm,fb);
  const t1=performance.now();
  const o=await sess.run({features:new ort.Tensor('float32',features,[1,80,frames]),lengths:new ort.Tensor('int64',BigInt64Array.from([BigInt(length)]),[1])});
  const t2=performance.now();
  const T=Number(o.out_lengths.data[0]),V=chars.length+1,lp=new Float32Array(o.log_probs.data.subarray(0,T*V));
  // greedy text
  let text='',prev=-1;
  for(let t=0;t<T;t++){let b=0;for(let v=1;v<V;v++)if(lp[t*V+v]>lp[t*V+b])b=v;if(b!==prev&&b!==V-1)text+=chars[b];prev=b;}
  text=text.replace(/ـ/g,'').trim();
  const el=$('timing'),sec=pcm.length/16000;
  if(el)el.textContent='⏱ صوت '+sec.toFixed(1)+'ث | تجهيز '+Math.round(t1-t0)+' | نموذج '+Math.round(t2-t1)+' ms | السرعة '+((t2-t0)/1000/sec).toFixed(2)+'× الزمن الحقيقي'+(self.crossOriginIsolated?' ⚡':'');
  return {text,lp,T};
}
function enqueue(pcm){
  queue=queue.then(async()=>{
    status('بيفحص…');
    try{const t=await transcribe(pcm);if(t&&t.text){heardParts.push(t.text);lpParts.push(t);}apply(finalized);}catch(e){status('خطأ: '+e.message);}
    status(live?'🎙️ بيسمع…':'جاهز ✔');
  });
  return queue;
}

// ---- recording ----
let ctx,stream,proc,src,buf=[],bufLen=0,speech=false,silentFor=0,recRate=48000;
function resample(f32,from,to){
  if(from===to) return f32;
  const r=from/to,n=Math.floor(f32.length/r),o=new Float32Array(n);
  for(let i=0;i<n;i++){const p=i*r,i0=Math.floor(p),i1=Math.min(i0+1,f32.length-1),fr=p-i0;o[i]=f32[i0]*(1-fr)+f32[i1]*fr;}
  return o;
}
function flush(){
  if(!bufLen) return;
  const all=new Float32Array(bufLen);let o=0;for(const b of buf){all.set(b,o);o+=b.length;}
  buf=[];bufLen=0;speech=false;silentFor=0;
  enqueue(resample(all,recRate,16000));
}
async function start(){
  try{stream=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:false,noiseSuppression:false,autoGainControl:true}});}
  catch(e){status('لازم تسمح بالميكروفون');return;}
  show2();
  ctx=new (window.AudioContext||window.webkitAudioContext)();recRate=ctx.sampleRate;
  src=ctx.createMediaStreamSource(stream);proc=ctx.createScriptProcessor(4096,1,1);
  proc.onaudioprocess=ev=>{
    const d=new Float32Array(ev.inputBuffer.getChannelData(0));
    let e=0;for(let i=0;i<d.length;i++)e+=d[i]*d[i];e=Math.sqrt(e/d.length);
    const dur=d.length/recRate;
    if(e>0.015){speech=true;silentFor=0;}else silentFor+=dur;
    if(speech||bufLen){buf.push(d);bufLen+=d.length;}
    if(speech&&silentFor>=0.45&&bufLen/recRate>1.0) flush();
    else if(bufLen/recRate>=12) flush();
  };
  src.connect(proc);proc.connect(ctx.destination);
  live=true;$('rec').textContent='⏹ وقّف';$('rec').classList.add('on');status('🎙️ بيسمع…');
}
function show2(){heardParts=[];lpParts=[];finalized=false;words.forEach(w=>{w.status='pending';w.heard='';});render(null);$('summary').textContent='';}
async function stop(){
  live=false;try{proc.disconnect();src.disconnect();stream.getTracks().forEach(t=>t.stop());await ctx.close();}catch{}
  flush();$('rec').textContent='🎙️ ابدأ التسميع';$('rec').classList.remove('on');
  status('بينهي الفحص…');
  await queue;finalized=true;apply(true);status('جاهز ✔');
}
$('rec').onclick=()=>live?stop():start();

// ---- audio file (for testing / checking a recording) ----
$('file').onchange=async ev=>{
  const f=ev.target.files[0];if(!f)return;
  show2();status('بيقرأ الملف…');
  const ac=new (window.AudioContext||window.webkitAudioContext)();
  const ab=await ac.decodeAudioData(await f.arrayBuffer());await ac.close();
  const oc=new OfflineAudioContext(1,Math.ceil(ab.duration*16000),16000),s=oc.createBufferSource();
  s.buffer=ab;s.connect(oc.destination);s.start();
  const pcm=(await oc.startRendering()).getChannelData(0);
  // split at silences (same idea as live mode): >=0.45 s quiet after >=1 s of audio, or at the quietest spot before 12 s
  const FR=1600,nf=Math.floor(pcm.length/FR),en=[];
  for(let f=0;f<nf;f++){let e=0;for(let k=0;k<FR;k+=4)e+=pcm[f*FR+k]*pcm[f*FR+k];en.push(Math.sqrt(e/(FR/4)));}
  let st=0,quiet=0;
  for(let f=0;f<nf;f++){
    quiet=en[f]<0.015?quiet+1:0;
    const len=(f+1-st);
    if((quiet>=5&&len>=10)||len>=120){
      let cut=f+1;
      if(len>=120&&quiet<5){let bv=1e9;for(let g=st+60;g<=f;g++){if(en[g]<bv){bv=en[g];cut=g+1;}}}
      enqueue(pcm.slice(st*FR,cut*FR));st=cut;quiet=0;f=cut-1;
    }
  }
  if(st*FR<pcm.length) enqueue(pcm.slice(st*FR));
  await queue;finalized=true;apply(true);status('جاهز ✔');ev.target.value='';
};

$('load').onclick=show;$('reset').onclick=show;$('hidden').onchange=()=>{localStorage.setItem('sel',JSON.stringify({s:+$('surah').value,f:+$('from').value,t:+$('to').value,h:$('hidden').checked}));render(null);};
$('surah').onchange=()=>{$('from').value=1;$('to').value=Math.min(5,Q[+$('surah').value].a.length);show();};
$('fast').textContent=FAST?'⚡ شغّال (اضغط للإلغاء)':'⚡ سريع (تجريبي)';
$('fast').onclick=async()=>{
  const base=location.pathname;
  if(FAST){location.href=base;return;}
  try{await navigator.serviceWorker.ready;}catch{}
  location.href=base+'?fast=1';
};
init();
