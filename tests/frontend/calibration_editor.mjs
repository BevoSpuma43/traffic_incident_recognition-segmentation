// Pure event/coordinate tests. These do not substitute for browser visual QA.
// Can run in any ES2022 runtime; supply the inline JS string to runEditorTests.
export function runEditorTests(JS) {
  const assert = (condition, message) => { if (!condition) throw new Error(message); };
  const results = [];
  const fixture = (overrides = {}) => {
    const events = {}, emitted = [], captured = new Set();
    const ctx = new Proxy({}, {get: (obj, key) => obj[key] ?? (() => {})});
    let rect = {left:100,top:50,width:960,height:540};
    let observer, disconnected=false;
    const canvas = {
      getContext: () => ctx, getBoundingClientRect: () => rect,
      setPointerCapture: id => captured.add(id), hasPointerCapture: id => captured.has(id),
      releasePointerCapture: id => captured.delete(id),
      addEventListener: (name,fn) => {events[name]=fn;},
      removeEventListener: name => {delete events[name];},
    };
    const data = {image_size:[1920,1080],vertices:[],roi_px:null,width_m:null,length_m:null,
      geometry_mode:'rectangle',mode:'calibration',max_points:4,readonly:false,
      record_id:'record',source_sha256:'hash',revision:3,epoch:7,image:'local',...overrides};
    class ImageMock { complete=false; naturalWidth=0; }
    class ObserverMock {
      constructor(callback) {observer=callback;}
      observe() {}
      disconnect() {disconnected=true;}
    }
    const factory = new Function('Image','ResizeObserver','crypto',
      JS.replace('export default function', 'return function'));
    const renderer = factory(ImageMock, ObserverMock, {randomUUID: () => 'event-'+(emitted.length+1)});
    const dispose = renderer({data,parentElement:{querySelector:s => s==='canvas' ? canvas : {}},
      setTriggerValue:(name,value) => emitted.push({name,...value})});
    const pointer = (type,x,y) => events[type]({button:0,pointerId:1,
      clientX:rect.left+x*rect.width/1920,clientY:rect.top+y*rect.height/1080,
      preventDefault() {}});
    const click = (x,y) => {pointer('pointerdown',x,y);pointer('pointerup',x,y);};
    return {emitted,events,pointer,click,dispose, resize: r => {rect=r;observer();},
      isDisconnected:() => disconnected};
  };
  const initial = [{id:'P1',x:100,y:100},{id:'P2',x:1800,y:100},
    {id:'P3',x:1800,y:900},{id:'P4',x:100,y:900}];
  {
    const f=fixture();
    for (const p of initial) f.click(p.x,p.y);
    assert(f.emitted.length===4,'one event per initialization click');
    assert(f.emitted[3].vertices.map(p=>p.id).join() === 'P1,P2,P3,P4','stable numbered IDs');
    f.click(950,500);
    assert(f.emitted.length===4,'rectangle does not accept a fifth point');
    results.push('four clicks and stable IDs');
  }
  {
    const f=fixture({vertices:initial});
    f.pointer('pointerdown',100,100);
    f.pointer('pointermove',200,200); f.pointer('pointermove',300,250);
    assert(f.emitted.length===0,'no backend events during drag');
    f.pointer('pointerup',300,250);
    assert(f.emitted.length===1,'one coherent event on release');
    assert(f.emitted[0].vertices[0].x===300 && f.emitted[0].vertices[0].y===250,'original pixel coordinates');
    assert(f.emitted[0].revision===3 && f.emitted[0].epoch===7,'revision and epoch bound event');
    f.resize({left:250,top:200,width:480,height:270});
    assert(f.emitted.length===1,'resize does not emit edits');
    f.pointer('pointerdown',300,250);f.pointer('pointermove',420,360);f.pointer('pointerup',420,360);
    assert(f.emitted[1].vertices[0].x===420 && f.emitted[1].vertices[0].y===360,'resized viewport with margins');
    results.push('release-only drag, resize and margins');
  }
  {
    const f=fixture({vertices:initial});
    f.pointer('pointerdown',100,100);f.pointer('pointermove',250,250);f.pointer('pointercancel',250,250);
    assert(f.emitted.length===0,'cancel does not save a drag');
    f.pointer('pointerdown',100,100);f.pointer('pointermove',3000,-1);f.pointer('pointerup',3000,-1);
    assert(f.emitted[0].vertices[0].x===1919 && f.emitted[0].vertices[0].y===0,'points clamped to video extent');
    f.dispose();
    assert(f.isDisconnected() && Object.keys(f.events).length===0,'listeners and observer disposed');
    results.push('cancel, bounds and cleanup');
  }
  {
    const f=fixture({vertices:initial,mode:'roi',roi_px:[]});
    f.click(200,200);f.click(1500,200);f.click(1500,700);
    assert(f.emitted[2].roi_px.length===3,'independent ROI vertices');
    assert(JSON.stringify(f.emitted[2].vertices)===JSON.stringify(initial),'ROI does not modify calibration');
    results.push('independent ROI');
  }
  {
    const f=fixture({vertices:initial,readonly:true});
    f.click(100,100);f.pointer('pointermove',200,200);
    assert(f.emitted.length===0,'confirmed editor is locked');
    results.push('confirmed editor lock');
  }
  return results;
}
