// CTC forced alignment scoring of expected words against model log-probs.
export const MARK=/^\p{Mn}$/u;
export function prep(w){return w.replace(/[\u06D6-\u06DB\u06E2\u06ED\s]/g,'').replace(/ـ/g,'').replace(/(.)([\u064B-\u0652\u0651]*)\u0670/gu,(m,l,mk)=>'وىا'.includes(l)?m:l+mk+'ـ\u0670').replace(/ْءَ/g,'ْـ\u0654\u064E');}
export function encodeWords(words,chars){
  const idx=new Map(chars.map((c,i)=>[c,i]));
  const seq=[],wordOf=[];
  words.forEach((w,wi)=>{
    if(wi>0){seq.push(idx.get(' '));wordOf.push(-1);}
    for(const ch of prep(w)){const k=idx.get(ch);if(k!==undefined){seq.push(k);wordOf.push(wi);}}
  });
  return {seq,wordOf};
}
// lp: Float32Array [T][V], blank=V-1. returns per-word {min, mean} of peak char logprob.
export function forcedScores(lp,T,V,words,chars){
  const {seq,wordOf}=encodeWords(words,chars);
  const L=seq.length,S=2*L+1,blank=V-1,NEG=-1e30;
  if(!L||T<1)return words.map(()=>({min:-99,mean:-99}));
  const lab=s=>(s&1)?seq[(s-1)>>1]:blank;
  let prev=new Float32Array(S).fill(NEG),cur=new Float32Array(S);
  const bp=new Uint8Array(T*S);
  prev[0]=lp[blank];if(S>1)prev[1]=lp[seq[0]];
  for(let t=1;t<T;t++){
    const o=t*V;
    for(let s=0;s<S;s++){
      let b=prev[s],k=0;
      if(s>0&&prev[s-1]>b){b=prev[s-1];k=1;}
      if(s>1&&(s&1)&&seq[(s-1)>>1]!==seq[(s-3)>>1]&&prev[s-2]>b){b=prev[s-2];k=2;}
      cur[s]=b+lp[o+lab(s)];bp[t*S+s]=k;
    }
    const tmp=prev;prev=cur;cur=tmp;
  }
  let s=(S>1&&prev[S-2]>prev[S-1])?S-2:S-1;
  const path=new Int32Array(T);
  for(let t=T-1;t>=0;t--){path[t]=s;s-=bp[t*S+s];}
  const peak=new Float32Array(L).fill(-99);
  for(let t=0;t<T;t++){const st=path[t];if(st&1){const li=(st-1)>>1;const v=lp[t*V+seq[li]];if(v>peak[li])peak[li]=v;}}
  const per=words.map(()=>[]);
  for(let i=0;i<L;i++)if(wordOf[i]>=0)per[wordOf[i]].push({ch:chars[seq[i]],p:peak[i]});
  return per.map(a=>{if(!a.length)return {min:-99,mean:-99,minLetter:-99,minMark:-99};
    const m=a.filter(x=>MARK.test(x.ch)),l=a.filter(x=>!MARK.test(x.ch));
    return {min:Math.min(...a.map(x=>x.p)),mean:a.reduce((q,x)=>q+x.p,0)/a.length,minLetter:l.length?Math.min(...l.map(x=>x.p)):0,minMark:m.length?Math.min(...m.map(x=>x.p)):0};});
}
