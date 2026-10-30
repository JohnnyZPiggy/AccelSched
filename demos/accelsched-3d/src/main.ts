import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { stateAt, type Bundle } from './replay';
import './style.css';

const app = document.querySelector<HTMLDivElement>('#app')!;
app.innerHTML = `
<header><a class="brand" href="https://github.com/JohnnyZPiggy/AccelSched">A<span>▰</span> ACCELSCHED</a><span class="edition">CLUSTER OBSERVATORY / 01</span><span class="badge">● SYNTHETIC REPLAY</span></header>
<main>
 <section class="stage" aria-label="Interactive 3D cluster">
  <div class="intro"><div class="eyebrow">HETEROGENEOUS COMPUTE · REAL DECISIONS</div><h1>Placement,<br><em>in motion.</em></h1><p>Explore the decisions behind the cluster.<br>Eight nodes. Three schedulers. One shared workload.</p></div>
  <div id="scene" aria-label="Orbit with drag, zoom with scroll or pinch, tap a rack to inspect"></div>
  <div id="scene-error" role="alert" hidden></div>
  <div id="event-tray" aria-label="Most recent scheduling event"></div>
  <div class="scene-caption"><span>01 / COMPUTE FLOOR</span><span>DRAG TO ORBIT · SCROLL TO ZOOM · TAP TO INSPECT</span></div>
  <div class="view-tools"><button id="home" title="Restore camera">↗ Home view</button><label><input id="quality" type="checkbox"> Low power</label></div>
 </section>
 <aside>
  <div class="eyebrow">SCHEDULING METHOD</div><label class="sr-only" for="method">Scheduling method</label><select id="method"><option value="first_fit">First Fit</option><option value="classic_best_fit">Classic Best Fit</option><option value="pgfp">PGFP</option></select>
  <div class="metrics"><div><strong id="accepted">—</strong><span>ACCEPTED</span></div><div><strong id="rejected">—</strong><span>REJECTED</span></div><div><strong id="errors">—</strong><span>SOLVER ERRORS</span></div></div>
  <p class="micro">Full replay totals · rejection is permanent</p>
  <div class="section-title"><span>LIVE ALLOCATION</span><span id="clock">t = 0</span></div><div class="gpu-count"><strong id="occupied">0</strong><span>/ 30 GPUs occupied</span></div><div class="total-track"><i id="total-fill"></i></div>
  <div class="section-title"><span>NODE INSPECTOR</span><span id="family"></span></div><div id="node-buttons" aria-label="Select a node"></div>
  <h2 id="node-title"></h2><div id="resources"></div><div class="job-label">RUNNING JOBS</div><div id="jobs"></div>
  <details><summary>Source & solver status</summary><p id="provenance"></p><p id="solver"></p><p>Heuristics carry no optimality certificate. Certificates, if any, apply only to an arrival batch. Host RAM is not VRAM. Time is in arbitrary fixture units.</p></details>
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
 const scene = new THREE.Scene(); scene.background = new THREE.Color('#080e13'); scene.fog = new THREE.FogExp2('#080e13', 0.022);
 const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
 function home() { camera.position.set(...(innerWidth<650?[22,17,26]:[15,12,18]) as [number,number,number]); controls?.target.set(0, 1.6, 0); controls?.update(); }
 let controls: OrbitControls | undefined;
 const reduced = matchMedia('(prefers-reduced-motion: reduce)');
 el<HTMLInputElement>('quality').checked = reduced.matches || innerWidth < 700;
 if (renderer) {
  renderer.setPixelRatio(Math.min(devicePixelRatio, el<HTMLInputElement>('quality').checked ? 1 : 1.5));
  renderer.outputColorSpace = THREE.SRGBColorSpace; renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.3;
  container.append(renderer.domElement);
  controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = !reduced.matches; controls.dampingFactor = 0.07; controls.minDistance = 9; controls.maxDistance = 45; controls.maxPolarAngle = Math.PI * 0.48;
 }
 home();
 scene.add(new THREE.HemisphereLight(0xc0deeb, 0x182721, 2.5));
 const key = new THREE.DirectionalLight(0xcce8ff, 4); key.position.set(5, 12, 8); scene.add(key);
 const rim = new THREE.DirectionalLight(0x56fbd0, 3); rim.position.set(-8, 5, -7); scene.add(rim);
 const metal = new THREE.MeshStandardMaterial({color: 0x303c46, metalness: 0.8, roughness: 0.32});
 const dark = new THREE.MeshStandardMaterial({color: 0x0c151c, metalness: 0.65, roughness: 0.5});
 const trim = new THREE.MeshStandardMaterial({color: 0x60727d, metalness: 0.9, roughness: 0.3});
 function box(parent: THREE.Object3D, x: number, y: number, z: number, w: number, h: number, d: number, material: THREE.Material) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(w,h,d), material); mesh.position.set(x,y,z); parent.add(mesh); return mesh;
 }
 box(scene,0,-0.23,0,16,0.4,12,metal);
 box(scene,0,-0.48,0,16.4,0.12,12.4,dark);
 const floor = new THREE.Mesh(new THREE.PlaneGeometry(200,200),new THREE.MeshStandardMaterial({color:0x080e13, metalness:0.4,roughness:0.5})); floor.rotation.x=-Math.PI/2; floor.position.y=-0.56; scene.add(floor);
 const grid = new THREE.GridHelper(16,16,0x354c52,0x1c2b33); grid.position.y=-0.015; scene.add(grid);
 const cyan = new THREE.MeshBasicMaterial({color:0x54ddda});
 for(const z of [-5.5,5.5]) box(scene,0,0.01,z,14,0.025,0.035,cyan);
 // Procedural racks are visual layout only; resource quantities come exclusively from reports.
 const racks = data.nodes.map((node,i) => {
  const group = new THREE.Group(); group.position.set((i%4-1.5)*3.35,0,(Math.floor(i/4)-0.5)*5); scene.add(group);
  box(group,0,0.15,0,2.3,0.3,2.4,dark); box(group,0,2.55,0,2.1,4.8,2.1,metal);
  box(group,0,2.55,1.07,1.88,4.5,0.07,dark);
  for(const x of [-0.98,0.98]) box(group,x,2.55,1.14,0.08,4.7,0.08,trim);
  const lights: THREE.MeshStandardMaterial[] = [];
  for(let slot=0;slot<8;slot++) {
   const y=0.65+slot*0.49;
   box(group,0,y,1.15,1.72,0.41,0.12,metal);
   for(let v=0;v<6;v++) box(group,-0.46+v*0.18,y,1.225,0.055,0.24,0.02,dark);
   const led = new THREE.MeshStandardMaterial({color:0x284747,emissive:0x284747,emissiveIntensity:0.4}); lights.push(led);
   box(group,0.72,y,1.23,0.075,0.2,0.04,led);
  }
  const labelCanvas = document.createElement('canvas'); labelCanvas.width=512;labelCanvas.height=128;
  const ctx=labelCanvas.getContext('2d')!; ctx.fillStyle='#101b23';ctx.fillRect(0,0,512,128);ctx.fillStyle='#dbebee';ctx.font='bold 52px monospace';ctx.fillText(node.id,24,63);ctx.fillStyle='#74c7bf';ctx.font='24px monospace';ctx.fillText(`FAMILY ${node.family} / ${node.capacity[0]} GPU`,24,104);
  const texture=new THREE.CanvasTexture(labelCanvas);texture.colorSpace=THREE.SRGBColorSpace;
  const label=new THREE.Mesh(new THREE.PlaneGeometry(1.8,0.45),new THREE.MeshBasicMaterial({map:texture}));label.position.set(0,4.7,1.12);group.add(label);
  const outline = new THREE.BoxHelper(group,0xa5ffd1); outline.visible=false; scene.add(outline);
  group.traverse(o => { o.userData.node=i; });
  return {group,lights,outline};
 });
 const ray = new THREE.Raycaster(); let down = {x:0,y:0};
 renderer?.domElement.addEventListener('pointerdown',e => {down={x:e.clientX,y:e.clientY};});
 renderer?.domElement.addEventListener('pointerup',e => {
  if(Math.hypot(e.clientX-down.x,e.clientY-down.y)>6) return;
  const rect=container.getBoundingClientRect(); ray.setFromCamera(new THREE.Vector2((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1),camera);
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
   rack.lights.forEach((led,s)=>{const on=s<Math.ceil(utilization*8);led.userData.target=new THREE.Color(on?(accepted?'#bdff88':'#4ce0cb'):(completed?'#6a99da':'#284747'));led.emissiveIntensity=on?2:0.3;});
   rack.outline.visible=i===selected;buttons[i].setAttribute('aria-pressed',String(i===selected));
  });
 }
 function setPlaying(value:boolean){playing=value;el('play').textContent=playing?'Ⅱ Pause':'▶ Play';el('play').setAttribute('aria-label',playing?'Pause replay':'Play replay');}
 el('play').addEventListener('click',()=>{if(time>=maxTime)time=0;setPlaying(!playing);update();});
 el('reset').addEventListener('click',()=>{time=0;setPlaying(false);update();});
 el('timeline').addEventListener('input',()=>{time=Number(el<HTMLInputElement>('timeline').value);setPlaying(false);update();});
 el('method').addEventListener('change',()=>{method=el<HTMLSelectElement>('method').value;update();});
 el('home').addEventListener('click',home);
 el('quality').addEventListener('change',()=>{renderer?.setPixelRatio(Math.min(devicePixelRatio,el<HTMLInputElement>('quality').checked?1:1.5));resize();});
 function resize(){const {width,height}=container.getBoundingClientRect();camera.aspect=width/height;camera.updateProjectionMatrix();renderer?.setSize(width,height);}
 new ResizeObserver(resize).observe(container);resize();update();
 let last=performance.now(), lastPaint=0;
 function frame(now:number){
  const dt=Math.min((now-last)/1000,0.1);last=now;
  if(playing&&!document.hidden){time=Math.min(maxTime,time+dt*Number(el<HTMLSelectElement>('speed').value)*3);if(time===maxTime)setPlaying(false);update();}
  if(!document.hidden && now-lastPaint>(el<HTMLInputElement>('quality').checked?1000/30:0)){racks.forEach(r=>r.lights.forEach(led=>{led.color.lerp(led.userData.target,reduced.matches?1:1-Math.exp(-(now-lastPaint)/140));led.emissive.copy(led.color);}));controls?.update();renderer?.render(scene,camera);lastPaint=now;}
  requestAnimationFrame(frame);
 }
 requestAnimationFrame(frame);
}
start().catch(error=>{el('scene-error').hidden=false;el('scene-error').textContent=`Unable to load replay: ${String(error)}`;el<HTMLButtonElement>('play').disabled=true;});
