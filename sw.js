const C="tasme3-v3";
const SHELL=['./','index.html','app.js','core.js','style.css','manifest.json','icon.svg','q0.json','q1.json','q2.json','q3.json','q4.json','q5.json','q6.json'];
self.addEventListener('install',e=>{e.waitUntil(caches.open(C).then(c=>c.addAll(SHELL)).then(()=>self.skipWaiting()));});
self.addEventListener('activate',e=>{e.waitUntil(caches.keys().then(k=>Promise.all(k.filter(x=>x!==C&&x.startsWith('tasme3')).map(x=>caches.delete(x)))).then(()=>self.clients.claim()));});
self.addEventListener('fetch',e=>{
  const u=new URL(e.request.url);
  if(u.origin!==location.origin&&u.hostname!=='cdn.jsdelivr.net') return; // model files are cached by transformers.js itself
  e.respondWith(caches.match(e.request).then(r=>r||fetch(e.request).then(res=>{if(u.origin!==location.origin&&res.ok){const c=res.clone();caches.open(C).then(ca=>ca.put(e.request,c));}return res;})));
});
