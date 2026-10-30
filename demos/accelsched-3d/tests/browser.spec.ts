import { test, expect } from '@playwright/test';
import { mkdirSync, writeFileSync } from 'node:fs';

for (const viewport of [{width:1440,height:1080},{width:390,height:844}]) {
 test(`replay interaction ${viewport.width}`,async({page})=>{
  await page.setViewportSize(viewport);
  const errors:string[]=[],requests:string[]=[],models:string[]=[];
  page.on('pageerror',e=>errors.push(e.message));
  page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
  page.on('request',r=>{if(r.url().endsWith('server-rack-light.glb'))models.push(r.url());if(!r.url().startsWith('http://127.0.0.1:5174/')&&!r.url().startsWith('blob:http://127.0.0.1:5174/')&&!r.url().startsWith('data:'))requests.push(r.url());});
  await page.goto('/');await expect(page.locator('#accepted')).toHaveText('22');
  await expect(page.locator('canvas')).toBeVisible();await expect(page.locator('#scene-error')).toBeHidden();
  await expect(page.locator('#scene')).toHaveAttribute('data-asset','grab3d-light');
  mkdirSync('evidence',{recursive:true});
  await expect(page.locator('.rack-tag')).toHaveCount(8);
  await page.waitForTimeout(300);
  if(viewport.width===1440)await page.screenshot({path:'evidence/default.png',fullPage:true});
  const tags=await page.locator('.rack-tag').all();
  let previousRight=0;
  for(let i=0;i<tags.length;i++){
   const tag=tags[i],bounds=(await tag.boundingBox())!;
   expect(bounds.x).toBeGreaterThanOrEqual(previousRight);previousRight=bounds.x+bounds.width;
   expect(previousRight).toBeLessThanOrEqual(viewport.width);
   const sceneBounds=(await page.locator('#scene canvas').boundingBox())!;
   // Every actual rack can be picked in the default view, without orbiting.
   await page.mouse.click(bounds.x+bounds.width/2,sceneBounds.y+sceneBounds.height*(viewport.width===390?.38:1)*.53);
   await expect(page.locator('#node-title')).toContainText(`N0${i+1}`);
  }
  if(viewport.width===1440)await page.screenshot({path:'evidence/selected.png',fullPage:true});
  for(const [method,total] of [['classic_best_fit','25'],['pgfp','21'],['first_fit','22']]){
   await page.selectOption('#method',method);await expect(page.locator('#accepted')).toHaveText(total);
  }
  const scrub=async(value:string)=>page.locator('#timeline').evaluate((e:HTMLInputElement,v)=>{e.value=v;e.dispatchEvent(new Event('input',{bubbles:true}));},value);
  await scrub('18');const forward=await page.locator('#occupied').textContent();
  await scrub('0');await scrub('18');await expect(page.locator('#occupied')).toHaveText(forward!);
  await page.getByRole('button',{name:'N04',exact:true}).click();await expect(page.locator('#node-title')).toContainText('N04');
  await page.locator('#play').click();await page.waitForTimeout(400);await page.locator('#play').click();
  const paused=await page.locator('#time').textContent();await page.waitForTimeout(200);await expect(page.locator('#time')).toHaveText(paused!);
  await page.locator('#reset').click();await expect(page.locator('#time')).toContainText('0.0 /');
  // Restore the default view before checking direct rack selection.
  await page.locator('#home').click();
  const canvas=page.locator('#scene canvas'), rect=(await canvas.boundingBox())!;
  let picked=false;
  for(const [x,y] of [[0.5,0.53],[0.65,0.53],[0.4,0.53],[0.6,0.53]]){
   await canvas.click({position:{x:rect.width*x,y:rect.height*y*(viewport.width===390?.38:1)}});
   if(!(await page.locator('#node-title').textContent())?.startsWith('N04')){picked=true;break;}
  }
  expect(picked).toBe(true);
  await page.mouse.move(rect.x+rect.width*.5,rect.y+rect.height*.65);await page.mouse.down();await page.mouse.move(rect.x+rect.width*.6,rect.y+rect.height*.6,{steps:10});await page.mouse.up();await page.mouse.wheel(0,-100);
  await page.locator('#home').click();await scrub('18');await page.getByRole('button',{name:'N01',exact:true}).click();await page.waitForTimeout(400);
  previousRight=0;
  for(const tag of tags){const bounds=(await tag.boundingBox())!;expect(bounds.x).toBeGreaterThanOrEqual(previousRight);previousRight=bounds.x+bounds.width;expect(previousRight).toBeLessThanOrEqual(viewport.width);}
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  mkdirSync('evidence',{recursive:true});
  await page.screenshot({path:`evidence/${viewport.width===390?'mobile':'desktop'}.png`,fullPage:true});
  await page.locator('#play').click();
  if(viewport.width===1440)await page.screenshot({path:'evidence/playback.png',fullPage:true});
  const intervals=await page.evaluate(()=>new Promise<number[]>(resolve=>{const a:number[]=[];let last=performance.now();function tick(t:number){a.push(t-last);last=t;if(a.length<61)requestAnimationFrame(tick);else resolve(a.slice(1));}requestAnimationFrame(tick);}));
  await page.locator('#play').click();
  writeFileSync(`evidence/browser-${viewport.width}.json`,JSON.stringify({viewport,errors,externalRequests:requests,frameIntervalsMs:intervals,note:'requestAnimationFrame cadence during playback in headless desktop Edge; not rendered GPU FPS or physical phone performance.'},null,2));
  expect(errors).toEqual([]);expect(requests).toEqual([]);expect(models).toHaveLength(1);
 });
}

test('reduced motion and WebGL fallback',async({page})=>{
 await page.emulateMedia({reducedMotion:'reduce'});await page.goto('/');await expect(page.locator('#accepted')).toHaveText('22');await expect(page.locator('#quality')).toBeChecked();await expect(page.locator('#play')).toHaveText('▶ Play');
 await page.addInitScript(()=>{const original=HTMLCanvasElement.prototype.getContext;HTMLCanvasElement.prototype.getContext=function(kind:string,...args:unknown[]){if(kind.startsWith('webgl'))return null;return original.apply(this,[kind,...args] as Parameters<typeof original>);} as typeof original;});
 await page.reload();await expect(page.locator('#scene-error')).toContainText('3D is unavailable');await page.getByRole('button',{name:'N08',exact:true}).click();await expect(page.locator('#node-title')).toContainText('N08');
});

test('touch controls and end of replay',async({browser})=>{
 const context=await browser.newContext({viewport:{width:390,height:844},isMobile:true,hasTouch:true});
 const page=await context.newPage();await page.goto('http://127.0.0.1:5174/');
 await page.getByRole('button',{name:'N06',exact:true}).tap();await expect(page.locator('#node-title')).toContainText('N06');
 await page.selectOption('#speed','2');await page.locator('#play').tap();await expect(page.locator('#play')).toHaveText('Ⅱ Pause');await page.locator('#play').tap();
 await page.locator('#timeline').evaluate((e:HTMLInputElement)=>{e.value=e.max;e.dispatchEvent(new Event('input',{bubbles:true}));});
 await expect(page.locator('#occupied')).toHaveText('0');await expect(page.locator('#jobs')).toHaveText('No jobs placed on this node');
 await page.locator('#reset').tap();await expect(page.locator('#time')).toContainText('0.0 /');
 await context.close();
});

test('model unavailable keeps replay usable',async({page})=>{
 await page.route('**/server-rack-light.glb',route=>route.abort());
 await page.goto('/');await expect(page.locator('#scene')).toHaveAttribute('data-asset','fallback');
 await expect(page.locator('#asset-note')).toContainText('Simple rack view');
 await page.getByRole('button',{name:'N03',exact:true}).click();await expect(page.locator('#node-title')).toContainText('N03');
 await page.selectOption('#method','pgfp');await expect(page.locator('#accepted')).toHaveText('21');
});
