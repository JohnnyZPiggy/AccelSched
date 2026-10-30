import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { stateAt, type Bundle } from './replay';
import './style.css';

const app = document.querySelector<HTMLDivElement>('#app')!;
app.innerHTML = `
<header><a class="brand" href="https://github.com/JohnnyZPiggy/AccelSched"><span class="brand-mark">a</span> AccelSched <span class="brand-divider">/</span><span class="product-name">Cluster replay</span></a><span class="badge">Synthetic workload · real decisions</span></header>
<main>
 <section class="stage" aria-label="Interactive 3D cluster">
  <div class="intro"><div class="eyebrow">The cluster, at a glance</div><h1>Room for every decision.</h1><p>Eight nodes. Three schedulers. Explore where each job lands.</p></div>
  <div id="scene" aria-label="Orbit with drag, zoom with scroll or pinch, tap a rack to inspect"></div>
  <div id="scene-error" role="alert" hidden></div>
  <div id="asset-note" role="status">Loading server model…</div>
  <div class="focus-caption"><span>Selected rack</span><strong id="focus-title">N01</strong><span>Illustrative hardware</span></div>
  <div id="event-tray" aria-label="Most recent scheduling event"></div>
  <div class="scene-caption"><span>Illustrative racks. Actual scheduling decisions.</span><span>Drag to orbit · pinch to zoom · tap to inspect</span></div>
  <div class="view-tools"><button id="home" title="Restore camera">↗ Home view</button><label><input id="quality" type="checkbox"> Low power</label></div>
 </section>
 <aside>
  <div class="eyebrow">Scheduling method</div><label class="sr-only" for="method">Scheduling method</label><select id="method"><option value="first_fit">First Fit</option><option value="classic_best_fit">Classic Best Fit</option><option value="pgfp">PGFP</option></select>
  <div class="metrics"><div><strong id="accepted">—</strong><span>Accepted</span></div><div><strong id="rejected">—</strong><span>Rejected</span></div><div><strong id="errors">—</strong><span>Errors</span></div></div>
  <p class="micro">Full replay totals · rejection is permanent</p>
  <div class="section-title"><span>GPU allocation</span><span id="clock">t = 0</span></div><div class="gpu-count"><strong id="occupied">0</strong><span>/ 30 GPUs occupied</span></div><div class="total-track"><i id="total-fill"></i></div>
  <div class="section-title"><span>Node details</span><span id="family"></span></div><div id="node-buttons" aria-label="Select a node"></div>
  <h2 id="node-title"></h2><div id="resources"></div><div class="job-label">Running jobs</div><div id="jobs"></div>
  <details><summary>About this replay</summary><p id="provenance"></p><p id="solver"></p><p>Heuristics carry no optimality certificate. Certificates, if any, apply only to an arrival batch. Host RAM is not VRAM. Time is in arbitrary fixture units.</p><p>Illustrative server art by Grab3D, CC0. Hardware appearance does not imply a real GPU model or vendor endorsement. <a href="./ASSET_CREDITS.md">Asset credits</a></p></details>
 </aside>
</main>
<footer>
 <div class="transport"><button id="play" aria-label="Play replay">▶ Play</button><button id="reset">↺ Reset</button><label class="speed">Speed <select id="speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label><output id="time">0 / 60</output></div>
 <label class="sr-only" for="timeline">Replay time</label><input id="timeline" type="range" min="0" max="60" step="0.1" value="0">
 <div class="legend"><span><i class="running"></i>Running</span><span><i class="accepted"></i>Newly accepted</span><span><i class="completed"></i>Completed</span><span><i class="rejected"></i>Rejected</span><span><i class="error"></i>Solver error</span><span id="event-status" aria-live="polite"></span></div>
</footer>`;
const el = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

async function start() {
 const response = await fetch('./data/replay.json');
 if (!response.ok) throw new Error(`Replay unavailable (${response.status})`);
 const data: Bundle = await response.json();
 if (data.schema_version !== 1 || !data.synthetic || Object.keys(data.reports).join() !== 'first_fit,classic_best_fit,pgfp') throw new Error('Unsupported replay schema');
 let method = 'first_fit', time = 0, playing = false, selected = 0;
 const maxTime = Math.max(...Object.values(data.reports).flatMap(r => r.events.map(e => e.time)));
 const totalGPU = data.nodes.reduce((s, n) => s + n.capacity[0], 0);
 el<HTMLInputElement>('timeline').max = String(maxTime);
 el('provenance').textContent = `Synthetic seed ${data.seed} · rolling.simulate · source ${data.source_sha ?? 'unavailable'}. Identical ordered inputs and 2 s solver limit for every method. No hardware telemetry or performance claims.`;
 const buttons = data.nodes.map((node, i) => {
  const b = document.createElement('button'); b.textContent = node.id; b.addEventListener('click', () => { selected = i; update(); }); el('node-buttons').append(b); return b;
 });
 const container = el('scene');
 let renderer: THREE.WebGLRenderer | undefined;
 try { renderer = new THREE.WebGLRenderer({antialias: true, powerPreference: 'low-power'}); }
 catch { el('scene-error').hidden = false; el('scene-error').textContent = '3D is unavailable on this device. The replay controls and node inspector remain available.'; }
 const scene = new THREE.Scene(); scene.background = new THREE.Color('#faf9f7');
 const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 250);
 let homeDistance=1;
 function home() {
  const distance = Math.max(12, layoutWidth / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * camera.aspect));
  // Clear orbit momentum before restoring the readable, front-facing row.
  if (controls) controls.enableDamping = false;
  controls?.update();
  controls?.target.set(0, 2.2, 0);
  camera.position.set(0, 2.2 + distance * 0.28, distance);
  homeDistance=distance*Math.hypot(1,.28);
  controls?.update();
  if (controls) controls.enableDamping = !reduced.matches;
 }
 let controls: OrbitControls | undefined;
 const reduced = matchMedia('(prefers-reduced-motion: reduce)');
 el<HTMLInputElement>('quality').checked = reduced.matches || innerWidth < 700;
 if (renderer) {
  renderer.setPixelRatio(Math.min(devicePixelRatio, el<HTMLInputElement>('quality').checked ? 1 : 1.5));
  renderer.outputColorSpace = THREE.SRGBColorSpace; renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.1;
  container.append(renderer.domElement);
  controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = !reduced.matches; controls.dampingFactor = 0.07; controls.minDistance = 9; controls.maxDistance = 130; controls.maxPolarAngle = Math.PI * 0.48;
 }
 scene.add(new THREE.HemisphereLight(0xffffff, 0x8d8882, 2));
 const key = new THREE.DirectionalLight(0xfffaf3, 2.3); key.position.set(5, 12, 8); scene.add(key);
 const rim = new THREE.DirectionalLight(0xe1e9f5, 1.2); rim.position.set(-8, 5, -7); scene.add(rim);
 function box(parent: THREE.Object3D, x: number, y: number, z: number, w: number, h: number, d: number, material: THREE.Material) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(w,h,d), material); mesh.position.set(x,y,z); parent.add(mesh); return mesh;
 }
 let template: THREE.Object3D;
 try {
  template = renderer ? (await new GLTFLoader().loadAsync('./assets/server-rack/server-rack-light.glb')).scene : new THREE.Group();
  if (!renderer) throw new Error('WebGL unavailable');
  // Source decoration is static art, never replay telemetry; overlays are separate.
  template.traverse(o=>{if(o instanceof THREE.Mesh && o.name.includes('led_glow'))o.visible=false;});
  container.dataset.asset='grab3d-light';el('asset-note').hidden=true;
 } catch {
  template=new THREE.Group();
  const shell=new THREE.MeshStandardMaterial({color:0xd3d4d4,roughness:.45,metalness:.2});
  const inset=new THREE.MeshStandardMaterial({color:0x43464b,roughness:.7});
  box(template,0,2.4,0,1.5,4.8,2,shell);
  for(let i=0;i<10;i++)box(template,0,.4+i*.43,1.03,1.32,.32,.08,inset);
  el('asset-note').textContent='Simple rack view · detailed model unavailable';container.dataset.asset='fallback';
 }
 const rawBounds=new THREE.Box3().setFromObject(template), rawSize=rawBounds.getSize(new THREE.Vector3());
 template.scale.setScalar(4.8/rawSize.y);
 const modelBounds=new THREE.Box3().setFromObject(template), center=modelBounds.getCenter(new THREE.Vector3());
 template.position.set(-center.x,-modelBounds.min.y,-center.z);
 const size=new THREE.Box3().setFromObject(template).getSize(new THREE.Vector3());
 const spacing=Math.max(size.x+size.z*.22+.7,2.5), layoutWidth=spacing*7+size.x+size.z*.3+1.2;
 const shadowCanvas=document.createElement('canvas');shadowCanvas.width=128;shadowCanvas.height=128;
 const shadowContext=shadowCanvas.getContext('2d')!, gradient=shadowContext.createRadialGradient(64,64,6,64,64,64);
 gradient.addColorStop(0,'rgba(32,29,24,0.24)');gradient.addColorStop(.45,'rgba(32,29,24,0.12)');gradient.addColorStop(1,'rgba(32,29,24,0)');shadowContext.fillStyle=gradient;shadowContext.fillRect(0,0,128,128);
 const shadowTexture=new THREE.CanvasTexture(shadowCanvas);
 // Shared geometry/materials/textures: one fetched GLB, eight lightweight clones.
 const racks = data.nodes.map((node,i) => {
  const group = new THREE.Group(); group.position.set((i-3.5)*spacing,0,0); scene.add(group);
  const art=template.clone(true);art.rotation.y=-.16;group.add(art);
  const contact=new THREE.Mesh(new THREE.PlaneGeometry(size.x+1.3,size.z+1.8),new THREE.MeshBasicMaterial({map:shadowTexture,transparent:true,depthWrite:false,toneMapped:false}));contact.rotation.x=-Math.PI/2;contact.position.y=-.015;group.add(contact);
  const lights: THREE.MeshStandardMaterial[] = [];
  for(let slot=0;slot<8;slot++) {
   const led = new THREE.MeshStandardMaterial({color:0xe2e3df,emissive:0xe2e3df,emissiveIntensity:0}); lights.push(led);
   box(group,0,.06,size.z/2+.32,(size.x*.8)/8,.045,.09,led).position.x=(slot-3.5)*size.x*.1;
  }
  const target=box(group,0,size.y/2,0,size.x+.35,size.y,size.z,new THREE.MeshBasicMaterial({visible:false}));
  group.updateMatrixWorld(true);
  const outline = new THREE.BoxHelper(target,0x738c80); outline.visible=false; scene.add(outline);
  const tag=document.createElement('button');tag.className='rack-tag';tag.textContent=node.id;tag.setAttribute('aria-label',`Inspect ${node.id}`);tag.addEventListener('click',()=>{selected=i;update();});container.append(tag);
  group.traverse(o => { o.userData.node=i; });
  return {group,lights,outline,tag};
 });
 const focusCamera=new THREE.PerspectiveCamera(32,1,.1,250);
 const ray = new THREE.Raycaster(); let down = {x:0,y:0};
 renderer?.domElement.addEventListener('pointerdown',e => {down={x:e.clientX,y:e.clientY};});
 renderer?.domElement.addEventListener('pointerup',e => {
  if(Math.hypot(e.clientX-down.x,e.clientY-down.y)>6) return;
  const rect=container.getBoundingClientRect(), height=innerWidth<650?rect.height*.38:rect.height;
  if(e.clientY-rect.top>height)return;
  ray.setFromCamera(new THREE.Vector2((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/height*2+1),camera);
  const hit=ray.intersectObjects(racks.map(r=>r.group),true)[0]; if(hit){selected=hit.object.userData.node;update();}
 });
 function update() {
  const report=data.reports[method], event=stateAt(report,time), node=data.nodes[selected];
  const load=event?.running_load ?? data.nodes.map(()=>[0,0,0]);
  const free=event?.free ?? data.nodes.map(n=>n.capacity);
  const recent=event && time-event.time<1.5;
  el('accepted').textContent=String(report.accepted_count);el('rejected').textContent=String(report.rejected_count);el('errors').textContent=String(report.solver_error_count);
  const occupied=load.reduce((s,n)=>s+n[0],0);el('occupied').textContent=String(occupied); el('total-fill').style.width=`${occupied/totalGPU*100}%`;
  el('clock').textContent=`t = ${time.toFixed(1)}`;el('time').textContent=`${time.toFixed(1)} / ${maxTime}`;el<HTMLInputElement>('timeline').value=String(time);
  el('node-title').textContent=`${node.id} / ${load[selected].some(v=>v>0)?'Active':'Idle'}`;el('family').textContent=`FAMILY ${node.family}`;
  el('focus-title').textContent=node.id;
  el('resources').innerHTML=['GPU count','CPU cores','Host RAM GiB'].map((label,k)=>`<div class="resource"><div><span>${label}</span><b>${load[selected][k]} <small>/ ${node.capacity[k]}</small></b></div><div class="track"><i style="width:${node.capacity[k]?load[selected][k]/node.capacity[k]*100:0}%"></i></div><div class="free">${free[selected][k]} free</div></div>`).join('');
  const active=event?.running.filter(r=>r.node===selected)??[];
  el('jobs').textContent=active.length?active.map(r=>`${r.job_id} → t${r.end}`).join(' · '):'No jobs placed on this node';
  el('event-status').textContent=event?`t${event.time} · +${event.accepted.length} accepted / ${event.completed.length} completed / ${event.rejected.length} rejected / ${event.solver_errors.length} errors`:'';
  const tray=el('event-tray');tray.replaceChildren();
  if(event) for(const [label,ids,cls] of [['Started',event.accepted,'accepted'],['Completed',event.completed,'completed'],['Rejected',event.rejected,'rejected'],['Solver error',event.solver_errors,'error']] as const){
   if(!ids.length)continue;
   const row=document.createElement('div');row.className=cls;
   row.textContent=`${label} / ${ids.join(' · ')}`;tray.append(row);
  }
  const batch=event?.batch_index!=null?report.batches[event.batch_index]:null;
  el('solver').textContent=batch?`Current batch: ${batch.solver_status} · validation ${batch.validation_status}${batch.error?` · ${batch.error}`:''}`:'Completion-only event; no solver call.';
  racks.forEach((rack,i)=>{
   const accepted=recent && event.accepted.some(j=>event.running.some(r=>r.job_id===j&&r.node===i));
   const completed=recent && event.completed.some(j=>report.decisions.find(d=>d.job_id===j)?.node===i);
   const utilization=Math.max(...load[i].map((v,k)=>data.nodes[i].capacity[k]?v/data.nodes[i].capacity[k]:0));
   rack.lights.forEach((led,s)=>{const on=s<Math.ceil(utilization*8);led.userData.target=new THREE.Color(on?(accepted?'#65b92e':'#16854c'):(completed?'#398a9c':'#b5c9bc'));led.emissiveIntensity=on?0.35:0;});
   rack.outline.visible=i===selected;buttons[i].setAttribute('aria-pressed',String(i===selected));rack.tag.setAttribute('aria-pressed',String(i===selected));
  });
 }
 function setPlaying(value:boolean){playing=value;el('play').textContent=playing?'Ⅱ Pause':'▶ Play';el('play').setAttribute('aria-label',playing?'Pause replay':'Play replay');}
 el('play').addEventListener('click',()=>{if(time>=maxTime)time=0;setPlaying(!playing);update();});
 el('reset').addEventListener('click',()=>{time=0;setPlaying(false);update();});
 el('timeline').addEventListener('input',()=>{time=Number(el<HTMLInputElement>('timeline').value);setPlaying(false);update();});
 el('method').addEventListener('change',()=>{method=el<HTMLSelectElement>('method').value;update();});
 el('home').addEventListener('click',home);
 el('quality').addEventListener('change',()=>{renderer?.setPixelRatio(Math.min(devicePixelRatio,el<HTMLInputElement>('quality').checked?1:1.5));resize();});
 function resize(){const {width,height}=container.getBoundingClientRect();camera.aspect=width/(innerWidth<650?height*.38:height);camera.updateProjectionMatrix();renderer?.setSize(width,height);home();}
 new ResizeObserver(resize).observe(container);resize();update();
 let last=performance.now(), lastPaint=0;
 function frame(now:number){
  const dt=Math.min((now-last)/1000,0.1);last=now;
  if(playing&&!document.hidden){time=Math.min(maxTime,time+dt*Number(el<HTMLSelectElement>('speed').value)*3);if(time===maxTime)setPlaying(false);update();}
  if(!document.hidden && now-lastPaint>(el<HTMLInputElement>('quality').checked?1000/30:0)){racks.forEach(r=>r.lights.forEach(led=>{led.color.lerp(led.userData.target,reduced.matches?1:1-Math.exp(-(now-lastPaint)/140));led.emissive.copy(led.color);}));controls?.update();
   const width=container.clientWidth,height=container.clientHeight,overviewHeight=innerWidth<650?height*.38:height;
   if(renderer){renderer.setScissorTest(true);renderer.setViewport(0,height-overviewHeight,width,overviewHeight);renderer.setScissor(0,height-overviewHeight,width,overviewHeight);renderer.render(scene,camera);
    if(innerWidth<650){const focusHeight=height-overviewHeight;racks.forEach((r,i)=>{r.group.visible=i===selected;r.outline.visible=i===selected;});focusCamera.aspect=width/focusHeight;focusCamera.updateProjectionMatrix();
     const focusTarget=new THREE.Vector3(racks[selected].group.position.x,2.3,0),offset=camera.position.clone().sub(controls?.target??new THREE.Vector3(0,2.2,0));
     const zoom=Math.min(1.8,Math.max(.65,offset.length()/homeDistance));offset.y*=.5;offset.normalize().multiplyScalar(9.8*zoom).applyAxisAngle(new THREE.Vector3(0,1,0),.32);
     focusCamera.position.copy(focusTarget).add(offset);focusCamera.lookAt(focusTarget);renderer.setViewport(0,0,width,focusHeight);renderer.setScissor(0,0,width,focusHeight);renderer.render(scene,focusCamera);racks.forEach((r,i)=>{r.group.visible=true;r.outline.visible=i===selected;});}
    renderer.setScissorTest(false);
   }
   racks.forEach(r=>{const p=new THREE.Vector3(r.group.position.x,size.y+.35,0).project(camera);r.tag.style.left=`${(p.x+1)*width/2}px`;r.tag.style.top=`${(1-p.y)*overviewHeight/2}px`;r.tag.hidden=!renderer||Math.abs(p.x)>1||Math.abs(p.y)>1||p.z>1;});lastPaint=now;}
  requestAnimationFrame(frame);
 }
 requestAnimationFrame(frame);
}
start().catch(error=>{el('scene-error').hidden=false;el('scene-error').textContent=`Unable to load replay: ${String(error)}`;el<HTMLButtonElement>('play').disabled=true;});
