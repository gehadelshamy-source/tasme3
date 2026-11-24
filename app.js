import {align,tokens,vowelMismatch} from './core.js';
import {pipeline, env} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2/dist/transformers.min.js';

const MODEL='nael239/tasme3-whisper-base-ar-quran-web';
const $=id=>document.getElementById(id);
const status=t=>$('status').textContent=t;
let Q=[], words=[], asr=null, asrP=null;
let heardParts=[], queue=Promise.resolve(), live=false, finalized=false;

env.allowLocalModels=false;
env.backends.onnx.wasm.numThreads=self.crossOriginIsolated?Math.min(4,navigator.hardwareConcurrency||2):1;

function loadModel(){
  if(asrP) return asrP;
  status('بنحمّل نموذج التعرف على الصوت (مرة واحدة فقط، حوالي ١٠٥ ميجا)…');
  asrP=pipeline('automatic-speech-recognition',MODEL,{dtype:'q8',device:'wasm',
    progress_callback:p=>{if(p.status==='progress'&&p.total>5e6)status('تحميل النموذج '+Math.round(p.progress)+'%');}})
    .then(m=>{asr=m;status('جاهز ✔ (بيشتغل على جهازك، الصوت مبيتبعتش لأي مكان)');return m;})
    .catch(e=>{asrP=null;status('فشل تحميل النموذج: '+e.message);throw e;});
  return asrP;
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
  heardParts=[];finalized=false;$('summary').textContent='';$('detail').hidden=true;
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
  words.forEach((w,i)=>{
    const x=r.words[i];
    if(!final&&i>last){w.status='pending';w.heard='';}
    else{w.status=x.status;w.heard=x.heard;
      if(x.status==='correct'&&vowelMismatch(w.text,x.heard)) w.status='tashkeel';}
  });
  const wrong=words.filter(w=>w.status==='wrong').length,close=words.filter(w=>w.status==='close').length,miss=words.filter(w=>w.status==='missing').length;
  const ok=words.filter(w=>w.status==='correct').length,tk=words.filter(w=>w.status==='tashkeel').length;
  $('summary').textContent=final?`النتيجة: ${r.score}% — صح ${ok} | قريب ${close} | تشكيل ${tk} | غلط ${wrong} | ناقص ${miss} | زيادة ${r.extras.length}`:`بتسمّع… (${ok} كلمة صح لحد دلوقتي)`;
  render(r);
}

async function transcribe(pcm){
  const m=await loadModel();
  let e=0;for(let i=0;i<pcm.length;i+=16)e+=pcm[i]*pcm[i];e/=pcm.length/16;
  if(e<1e-6) return '';
  const t0=performance.now();
  const fe=m.processor.feature_extractor;
  const ns=Math.min(480000,Math.max(160000,Math.ceil(pcm.length/80000)*80000));
  fe.config.n_samples=ns;fe.config.nb_max_frames=ns/160;
  const f=(await fe(pcm)).input_features;
  const t1=performance.now();
  const out=await m.model.generate({inputs:f,num_beams:1,max_new_tokens:Math.min(200,Math.ceil(pcm.length/16000*8)+12)});
  const t2=performance.now();
  const text=m.tokenizer.decode(out.tolist?out.tolist()[0]:out[0],{skip_special_tokens:true}).trim();
  const el=document.getElementById('timing');
  if(el)el.textContent='⏱ صوت '+(pcm.length/16000).toFixed(1)+'ث | تجهيز '+Math.round(t1-t0)+' | فك '+Math.round(t2-t1)+' ms';
  return text;
}
function enqueue(pcm){
  queue=queue.then(async()=>{
    status('بيفحص…');
    try{const t=await transcribe(pcm);if(t.trim())heardParts.push(t);apply(finalized);}catch(e){status('خطأ: '+e.message);}
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
    if(speech&&silentFor>=0.7&&bufLen/recRate>1.2) flush();
    else if(bufLen/recRate>=20) flush();
  };
  src.connect(proc);proc.connect(ctx.destination);
  live=true;$('rec').textContent='⏹ وقّف';$('rec').classList.add('on');status('🎙️ بيسمع…');
}
function show2(){heardParts=[];finalized=false;words.forEach(w=>{w.status='pending';w.heard='';});render(null);$('summary').textContent='';}
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
  const N=16000*20;
  for(let i=0;i<pcm.length;){
    let end=Math.min(pcm.length,i+N);
    if(end<pcm.length){ // cut at quietest 0.1s window in last 4s
      let best=end,bv=1e9;for(let p=end-64000;p<end-1600;p+=800){let e=0;for(let k=0;k<1600;k+=8)e+=Math.abs(pcm[p+k]);if(e<bv){bv=e;best=p+800;}}end=best;}
    enqueue(pcm.slice(i,end));i=end;
  }
  await queue;finalized=true;apply(true);status('جاهز ✔');ev.target.value='';
};

$('load').onclick=show;$('reset').onclick=show;$('hidden').onchange=()=>{localStorage.setItem('sel',JSON.stringify({s:+$('surah').value,f:+$('from').value,t:+$('to').value,h:$('hidden').checked}));render(null);};
$('surah').onchange=()=>{$('from').value=1;$('to').value=Math.min(5,Q[+$('surah').value].a.length);show();};
init();
