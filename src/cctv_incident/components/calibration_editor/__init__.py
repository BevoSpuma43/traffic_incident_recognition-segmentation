"""Inline CCv2 canvas. All frontend code is local; no CDN, iframe or build step."""

import streamlit as st

HTML = """
<div class="editor">
  <canvas aria-label="Editor punti di calibrazione sul primo frame" tabindex="0"></canvas>
  <p class="hint" aria-live="polite"></p>
</div>
"""

CSS = """
.editor { width:100%; max-width:1280px; }
canvas { display:block; width:100%; height:auto; touch-action:none;
  outline:1px solid var(--st-border-color,#888); border-radius:6px; cursor:crosshair; }
.hint { margin:6px 0; color:var(--st-text-color); font-size:0.9rem; }
"""

JS = """
export default function(component) {
  const {parentElement, data, setTriggerValue} = component;
  const canvas = parentElement.querySelector('canvas');
  const hint = parentElement.querySelector('.hint');
  const ctx = canvas.getContext('2d');
  const W = data.image_size[0], H = data.image_size[1];
  canvas.width = W; canvas.height = H;
  let vertices = data.vertices.map(p => ({...p}));
  let roi = data.roi_px === null ? [[0,0],[W-1,0],[W-1,H-1],[0,H-1]]
                              : data.roi_px.map(p => [...p]);
  let active = null, activePointer = null, before = null, dirty = false;
  let alive = true;
  const image = new Image();
  const ratio = () => W / Math.max(1, canvas.getBoundingClientRect().width);
  const position = event => {
    const rect = canvas.getBoundingClientRect();
    return [Math.max(0, Math.min(W-1, (event.clientX-rect.left)*W/rect.width)),
            Math.max(0, Math.min(H-1, (event.clientY-rect.top)*H/rect.height))];
  };
  const points = () => data.mode === 'roi' ? roi : vertices.map(p => [p.x,p.y]);
  function polygon(list, color, dashed) {
    if (!list.length) return;
    ctx.strokeStyle=color; ctx.lineWidth=2*ratio();
    ctx.setLineDash(dashed ? [7*ratio(),5*ratio()] : []);
    ctx.beginPath(); list.forEach((p,i) => i ? ctx.lineTo(...p) : ctx.moveTo(...p));
    if (list.length >= 3) ctx.closePath();
    ctx.stroke(); ctx.setLineDash([]);
  }
  function draw() {
    if (!alive) return;
    ctx.clearRect(0,0,W,H);
    if (image.complete && image.naturalWidth) ctx.drawImage(image,0,0,W,H);
    polygon(roi,'#45b8ff',true);
    polygon(vertices.map(p=>[p.x,p.y]),'#ffda45',false);
    const r=ratio();
    const handles = vertices.map(p=>({...p,color:'#ffda45'}));
    if (data.mode==='roi') handles.push(...roi.map((p,i)=>({x:p[0],y:p[1],id:'R'+(i+1),color:'#45b8ff'})));
    for (const p of handles) {
      ctx.fillStyle=p.color;
      ctx.beginPath(); ctx.arc(p.x,p.y,7*r,0,Math.PI*2); ctx.fill();
      ctx.strokeStyle='#151515'; ctx.lineWidth=2*r; ctx.stroke();
      ctx.font=`bold ${14*r}px sans-serif`; ctx.lineWidth=4*r;
      ctx.strokeText(p.id,p.x+10*r,p.y-10*r); ctx.fillText(p.id,p.x+10*r,p.y-10*r);
    }
    const indexed = Object.fromEntries(vertices.map(p=>[p.id,p]));
    if (data.geometry_mode === 'rectangle' && vertices.length === 4) {
      for (const [a,b,label] of [['P1','P2','Larghezza'],['P2','P3','Lunghezza']]) {
        const p=indexed[a], q=indexed[b];
        const value=label==='Larghezza' ? data.width_m : data.length_m;
        const text=label+(value===null ? ' (m)' : ' '+value+' m');
        ctx.font=`bold ${13*r}px sans-serif`; ctx.strokeStyle='#151515';
        ctx.fillStyle='#ffda45'; ctx.lineWidth=4*r;
        ctx.strokeText(text,(p.x+q.x)/2,(p.y+q.y)/2-8*r);
        ctx.fillText(text,(p.x+q.x)/2,(p.y+q.y)/2-8*r);
      }
    }
    hint.textContent = data.readonly ? 'Calibrazione salvata: premi Modifica calibrazione per correggerla.'
      : data.mode==='roi' ? 'ROI azzurra indipendente: trascina i vertici o clicca per aggiungerli.'
      : vertices.length<4 ? 'Clicca i vertici del rettangolo in ordine P1, P2, P3, P4.'
      : 'Trascina i punti gialli; le coordinate si aggiornano al rilascio.';
  }
  const down = event => {
    if (data.readonly || event.button!==0 || active!==null) return;
    const pos=position(event), list=points();
    before={vertices:vertices.map(p=>({...p})),roi:roi.map(p=>[...p])};
    active=list.findIndex(p=>Math.hypot(p[0]-pos[0],p[1]-pos[1])<=12*ratio());
    dirty=false;
    if (active<0) {
      if (data.mode==='roi' && roi.length<32) { roi.push(pos); active=roi.length-1; dirty=true; }
      else if (data.mode==='calibration' && vertices.length<data.max_points) {
        let index=1; while(vertices.some(p=>p.id==='P'+index)) index++;
        vertices.push({id:'P'+index,x:pos[0],y:pos[1]}); active=vertices.length-1; dirty=true;
      } else { active=null; return; }
    }
    activePointer=event.pointerId;
    canvas.setPointerCapture(event.pointerId); event.preventDefault(); draw();
  };
  const move = event => {
    if (active===null || event.pointerId!==activePointer) return;
    const pos=position(event);
    if (data.mode==='roi') roi[active]=pos;
    else {vertices[active].x=pos[0];vertices[active].y=pos[1];}
    dirty=true; draw();
  };
  const up = event => {
    if (active===null || event.pointerId!==activePointer) return;
    if (dirty) {
      const pos=position(event);
      if (data.mode==='roi') roi[active]=pos;
      else {vertices[active].x=pos[0];vertices[active].y=pos[1];}
    }
    active=null;activePointer=null;
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    if (dirty) setTriggerValue('edit', {
      record_id:data.record_id, source_sha256:data.source_sha256, revision:data.revision,
      epoch:data.epoch, event_id:crypto.randomUUID?.() ?? Date.now()+'-'+Math.random(), mode:data.mode,
      vertices:vertices.map(p=>({...p})),
      roi_px:data.mode==='roi' ? roi : data.roi_px
    });
    dirty=false;
  };
  const cancel = event => {
    if (active===null || event.pointerId!==activePointer) return;
    if (before) {vertices=before.vertices;roi=before.roi;}
    active=null;activePointer=null;dirty=false;
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    draw();
  };
  canvas.addEventListener('pointerdown',down);
  canvas.addEventListener('pointermove',move);
  canvas.addEventListener('pointerup',up);
  canvas.addEventListener('pointercancel',cancel);
  const observer = new ResizeObserver(draw); observer.observe(canvas);
  image.onload=draw; image.src=data.image;
  draw();
  return () => {
    if (activePointer!==null && canvas.hasPointerCapture(activePointer)) canvas.releasePointerCapture(activePointer);
    alive=false;observer.disconnect();image.onload=null;
    canvas.removeEventListener('pointerdown',down);canvas.removeEventListener('pointermove',move);
    canvas.removeEventListener('pointerup',up);canvas.removeEventListener('pointercancel',cancel);
  };
}
"""

_COMPONENT = st.components.v2.component("road_calibration_editor", html=HTML, css=CSS, js=JS)


def calibration_editor(*, data, key, on_edit):
    return _COMPONENT(data=data, key=key, height="content", on_edit_change=on_edit)
