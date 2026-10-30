import { chromium } from '@playwright/test';
import assert from 'node:assert/strict';
import { writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';

const url='https://laptop-dv0lktef.tail9ceb46.ts.net:8443/';
const browser=await chromium.launch({channel:'msedge'});
try {
 const context=await browser.newContext({viewport:{width:390,height:844},isMobile:true,hasTouch:true});
 const page=await context.newPage(), errors=[], failed=[], modelRequests=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
 page.on('requestfailed',r=>failed.push({url:r.url(),error:r.failure()?.errorText}));
 page.on('request',r=>{if(r.url().endsWith('server-rack-light.glb'))modelRequests.push(r.url());});
 const response=await page.goto(url,{waitUntil:'networkidle',timeout:60000});assert.equal(response.status(),200);
 await page.locator('#scene[data-asset="grab3d-light"]').waitFor();
 assert.equal(await page.locator('#scene-error').isVisible(),false);
 assert.equal(await page.locator('#scene canvas').count(),1);assert.equal(await page.locator('.rack-tag').count(),8);
 assert.equal(await page.locator('#accepted').textContent(),'22');
 await page.getByRole('button',{name:'Inspect N06',exact:true}).tap();assert.match(await page.locator('#node-title').textContent(),/N06/);
 assert.equal(await page.locator('#focus-title').textContent(),'N06');
 await page.selectOption('#method','classic_best_fit');assert.equal(await page.locator('#accepted').textContent(),'25');
 await page.locator('#play').tap();await page.waitForTimeout(500);await page.locator('#play').tap();assert.notEqual(await page.locator('#time').textContent(),'0.0 / 54');
 await page.locator('#timeline').evaluate(e=>{e.value='18';e.dispatchEvent(new Event('input',{bubbles:true}));});
 await page.waitForTimeout(300);await page.screenshot({path:'evidence/mobile-https.png',fullPage:true});
 const asset=await context.request.get(new URL('assets/server-rack/server-rack-light.glb',url).href);
 const assetBytes=await asset.body(), hash=createHash('sha256').update(assetBytes).digest('hex');
 assert.equal(asset.status(),200);assert.equal(hash,'246602b854778930ea71fd612e91de6b71b809f8d31de9d261c44a94223c2139');
 const good=await context.request.get('http://127.0.0.1:5174/',{headers:{Host:'laptop-dv0lktef.tail9ceb46.ts.net'}});assert.equal(good.status(),200);
 const bad=await context.request.get('http://127.0.0.1:5174/',{headers:{Host:'untrusted.example'}});assert.equal(bad.status(),403);
 const json=await context.request.get(new URL('data/replay.json',url).href);assert.equal(json.status(),200);assert.equal((await json.json()).schema_version,1);
 assert.equal(modelRequests.length,1);assert.deepEqual(errors,[]);assert.deepEqual(failed,[]);
 const result={url,httpStatus:response.status(),certificateValidation:'enabled; no ignoreHTTPSErrors',browser:await browser.version(),viewport:{width:390,height:844},assetBytes:assetBytes.length,assetSHA256:hash,modelFetchCount:modelRequests.length,checks:['real GLB loaded','eight node labels','touch selection and focus N06','method switch','play/pause','scrub','JSON via HTTPS','exact Host 200','untrusted Host 403'],errors,failed,physicalIPhoneSafari:'NOT_RUN; Edge touch emulation on Windows'};
 writeFileSync('evidence/mobile-https.json',JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result,null,2));
}finally{await browser.close();}
