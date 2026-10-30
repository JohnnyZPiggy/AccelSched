// Bounded local comparison of the publisher's two GLBs; no runtime dependency.
import { chromium } from '@playwright/test';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
const browser = await chromium.launch({channel:'msedge'});
try {
 mkdirSync('test-results',{recursive:true});
 if(!existsSync('test-results/server-rack-full.glb')){
  const response=await fetch('https://grab3d.com/dl/server-rack/server-rack.glb');
  if(!response.ok||Number(response.headers.get('content-length'))>2_000_000)throw new Error('Unexpected full model response');
  const bytes=new Uint8Array(await response.arrayBuffer());if(bytes.length>2_000_000)throw new Error('Model exceeds audit bound');
  writeFileSync('test-results/server-rack-full.glb',bytes);
 }
 const page = await browser.newPage({viewport:{width:800,height:800}});
 await page.route('**/asset-audit', route => route.fulfill({contentType:'text/html',body:'<html><body style="margin:0"><script type="importmap">{"imports":{"three":"/node_modules/three/build/three.module.js"}}</script></body></html>'}));
 await page.goto('http://127.0.0.1:5174/asset-audit');
 const results=[];
 for(const [variant,path] of [['light','public/assets/server-rack/server-rack-light.glb'],['full','test-results/server-rack-full.glb']]){
  const bytes=readFileSync(path), json=JSON.parse(bytes.subarray(20,20+bytes.readUInt32LE(12)).toString());
  const triangles=json.meshes.reduce((sum,m)=>sum+m.primitives.reduce((s,p)=>s+json.accessors[p.indices??p.attributes.POSITION].count/3,0),0);
  const result=await page.evaluate(async url=>{
   const THREE=await import('/node_modules/three/build/three.module.js');
   const {GLTFLoader}=await import('/node_modules/three/examples/jsm/loaders/GLTFLoader.js');
   const model=(await new GLTFLoader().loadAsync(url)).scene;
   const bounds=new THREE.Box3().setFromObject(model),size=bounds.getSize(new THREE.Vector3()),center=bounds.getCenter(new THREE.Vector3());
   model.position.sub(center);
   const scene=new THREE.Scene();scene.background=new THREE.Color('#f7f7f5');scene.add(model);
   scene.add(new THREE.HemisphereLight(0xffffff,0x757575,2));const light=new THREE.DirectionalLight(0xffffff,2.5);light.position.set(3,5,6);scene.add(light);
   const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setSize(800,800);renderer.setPixelRatio(1);renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1;
   document.body.replaceChildren(renderer.domElement);const camera=new THREE.PerspectiveCamera(34,1,.01,100);camera.position.set(2.3,1.4,4.5);camera.lookAt(0,0,0);renderer.render(scene,camera);
   const meshes=[],textures=new Map(),arrays=new Set();let geometryBytes=0;
   model.traverse(o=>{if(o.isMesh){meshes.push({name:o.name,material:Array.isArray(o.material)?o.material.map(m=>m.name):o.material.name});for(const a of [...Object.values(o.geometry.attributes),o.geometry.index].filter(Boolean)){if(!arrays.has(a.array)){arrays.add(a.array);geometryBytes+=a.array.byteLength;}}for(const m of Array.isArray(o.material)?o.material:[o.material])for(const t of Object.values(m)){if(t?.isTexture&&!textures.has(t.uuid))textures.set(t.uuid,{width:t.image.width,height:t.image.height});}}});
   window.auditRenderer=renderer;window.auditScene=scene;
   return {bounds:size.toArray(),meshes,textureDimensions:[...textures.values()],estimatedRGBAWithMipBytes:[...textures.values()].reduce((s,t)=>s+t.width*t.height*4*4/3,0),geometryAttributeBytes:geometryBytes,render:renderer.info.render};
  },'/'+path.replace(/^public\//,''));
  await page.screenshot({path:`evidence/assets/grab3d-${variant}.png`});
  results.push({variant,bytes:bytes.length,sha256:createHash('sha256').update(bytes).digest('hex'),triangles,images:json.images?.length??0,textures:json.textures?.length??0,materials:json.materials.length,...result});
  await page.evaluate(()=>{window.auditRenderer.dispose();window.auditScene.traverse(o=>{if(o.isMesh){o.geometry.dispose();for(const m of Array.isArray(o.material)?o.material:[o.material])m.dispose();}});});
 }
 writeFileSync('evidence/assets/glb-audit.json',JSON.stringify(results,null,2)+'\n');
 console.log(JSON.stringify(results.map(({meshes,...r})=>r),null,2));
}finally{await browser.close();}
