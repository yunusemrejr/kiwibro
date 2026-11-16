# three.js reference

## three.js, a 3D library for the browser
three.js draws real 3D in a canvas with WebGL: you build a scene of objects, a camera looks at it, and a renderer draws each frame. It is a JavaScript library loaded in the page from a CDN module URL, with no build step. Its units are yours to choose; most examples treat one unit as a metre and y as up. Everything visible is a Mesh: a geometry (the shape) plus a material (the surface).

## three.js a minimal page has five parts
A working three.js page has a renderer, a scene, a camera, at least one light, and a loop that renders every frame; a resize handler keeps it sharp when the window changes. The renderer's canvas goes into the page: document.body.append(renderer.domElement). The camera starts at the origin, inside the scene, so move it back (camera.position.set(0, 2, 6)) and aim it with camera.lookAt(0, 0, 0).

## three.js import map for the CDN
In the head, before any module script: <script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.min.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/"}}</script>. Then import * as THREE from "three"; and import { OrbitControls } from "three/addons/controls/OrbitControls.js". The map lets an addon file find "three" itself. It must come before the first module script, or the browser ignores it.

## three.js renderer setup
const renderer = new THREE.WebGLRenderer({ antialias: true }); renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); renderer.setSize(innerWidth, innerHeight); document.body.append(renderer.domElement); Capping the pixel ratio at 2 stops a phone with a 3x screen from drawing nine times the pixels. Antialias smooths the jagged edges of geometry; it costs a little on large screens and is worth it for most scenes.

## three.js scene
const scene = new THREE.Scene(); sets the world. scene.background = new THREE.Color(0x101820) is the clear colour behind everything, and scene.fog = new THREE.Fog(0x101820, 10, 60) fades distant objects into it. Add objects with scene.add(mesh) and take them out with scene.remove(mesh). An object's place is its position: mesh.position.set(x, y, z); rotation.y += 0.01 per frame spins it about the up axis.

## three.js PerspectiveCamera and framing
new THREE.PerspectiveCamera(fov, aspect, near, far), for example (50, innerWidth / innerHeight, 0.1, 100). fov is the vertical view in degrees, and 50 to 60 looks natural; near and far are clipping distances, so nothing closer than 0.1 or further than 100 is drawn. Move the camera back far enough that the subject fits: with fov 50 and a two-unit object, z = 6 is a good start. camera.lookAt(0, 0, 0) points it at the centre.

## three.js light, or the scene is black
A MeshStandardMaterial takes its colour from the light around it and renders black with none. Use a hemisphere light, sky above and ground below: scene.add(new THREE.HemisphereLight(0xffffff, 0x445566, 2)); or a directional key light plus an ambient fill: const key = new THREE.DirectionalLight(0xffffff, 3); key.position.set(5, 8, 4); scene.add(key, new THREE.AmbientLight(0xffffff, 0.6)). Recent three.js uses physical intensities, so 1 to 3 is normal, not 0.1.

## three.js animation loop
renderer.setAnimationLoop(() => { mesh.rotation.y += 0.01; renderer.render(scene, camera); }); asks the browser for a frame and repeats itself, about 60 times a second, and stops while the tab is hidden. Change at least one thing per frame or the loop shows nothing new. renderer.setAnimationLoop(null) stops it. For motion in real time, use a clock: const clock = new THREE.Clock(); const dt = clock.getDelta(); then mesh.rotation.y += dt;

## three.js resize handler
addEventListener("resize", () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); }); All three lines matter: without updateProjectionMatrix the picture stretches, and without the new size the canvas keeps its old pixels and looks blurry. A canvas inside a sized container uses container.clientWidth and clientHeight in the same three lines.

## three.js forgetting to render
A page with a renderer, scene and camera but no render call shows a blank canvas. renderer.render(scene, camera) draws one frame and must be called again for each later frame, which is what setAnimationLoop is for. A still image needs one call and no loop. When the canvas stays blank, read the browser console: a module or shader error, or a camera pointing away from everything, looks exactly like nothing drawn.

## three.js OrbitControls, the right import path
import { OrbitControls } from "three/addons/controls/OrbitControls.js"; const controls = new OrbitControls(camera, renderer.domElement); gives mouse and touch orbit, pan and zoom. The path is wrong when it points at three/examples/jsm/..., at a local copy that is not there, or at a path relative to the HTML file: the string starts with three/addons/ and the import map must map it. With controls.enableDamping = true, call controls.update() in the loop.

## three.js local modules do not load from file://
A page opened by double-clicking it has the address file://..., and the browser refuses ES module imports there, along with many texture and model fetches. An inline module script and the CDN import map still work, because they are not local files; <script type="module" src="main.js"> and local .js files do not. Either keep all the code inline in the HTML, or serve the folder from a local web server.

## three.js geometry and mesh
A mesh is geometry plus material: new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshStandardMaterial({ color: 0x4fa34f })). BoxGeometry(width, height, depth), SphereGeometry(radius, widthSegments, heightSegments), PlaneGeometry, CylinderGeometry, ConeGeometry, TorusGeometry and TorusKnotGeometry are built in. A plane lies in the xy plane, so ground.rotation.x = -Math.PI / 2 lays it flat. geometry.rotateX, translate and scale change the shape before it is drawn.

## three.js material choice
MeshStandardMaterial is the physically based default: it needs light and takes roughness (0 is a mirror, 1 is matte) and metalness. MeshBasicMaterial ignores light and shows one flat colour, which suits lines, markers and debug helpers but looks flat on objects. MeshNormalMaterial and MeshDepthMaterial are for debugging. A material colour is a hex number (0xff8844) or a CSS string ("#ff8844"); change it with material.color.set(0x3366ff).

## three.js lights and shadows
Shadows are off by default: renderer.shadowMap.enabled = true; light.castShadow = true; then mesh.castShadow = true and ground.receiveShadow = true. Only DirectionalLight, SpotLight and PointLight cast shadows; HemisphereLight and AmbientLight do not. A directional light's shadow camera is small by default, so set light.shadow.camera.left, right, top and bottom to cover the scene, or the shadow is cut off or very blurry.

## three.js loading a GLB model
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js"; new GLTFLoader().load("model.glb", gltf => { scene.add(gltf.scene); }); loads a Blender or glTF export. The path is relative to the page, and the page must be served over http, not file://. Geometry compressed with Draco or Meshopt needs DRACOLoader or MeshoptDecoder set on the loader. Give the load call an error callback: a missing file fails silently otherwise.

## three.js textures and colour space
A texture that carries colour (a photo, a painted map) must be marked sRGB: const map = new THREE.TextureLoader().load("wood.jpg"); map.colorSpace = THREE.SRGBColorSpace; while roughness, normal and metalness maps stay linear. Recent three.js outputs sRGB by default, so a colour texture left linear looks washed out. Textures load asynchronously, so the first frames show the material without them.

## three.js performance on a laptop
Cap the pixel ratio at 2, keep draw calls low by reusing one geometry and material for repeated objects (or an InstancedMesh), and avoid creating objects inside the animation loop. A few thousand triangles at 60 fps is fine on an integrated GPU; large shadows, big textures and post-processing are what slow it down. Test the page at phone width and with the window made small and large before calling it smooth.

## three.js dispose and cleanup
Every geometry, material and texture holds GPU memory the JavaScript garbage collector cannot free. Removing a mesh from the scene stops drawing it but keeps that memory: call mesh.geometry.dispose() and dispose() on each of its materials too, and texture.dispose() on its maps. renderer.setAnimationLoop(null) stops the loop and renderer.dispose() frees the context. A page that switches scenes without disposing leaks until the tab closes.

## three.js the canvas and the page layout
html, body { margin: 0; height: 100%; } body { overflow: hidden; } and canvas { display: block; } stop a scrollbar and the thin gap under the canvas that its inline text baseline leaves. innerWidth and innerHeight include the scrollbar and are right for a full-window canvas. To fill only a box, set the renderer size to the box's clientWidth and clientHeight and the camera aspect to their ratio.

## three.js colour and tone mapping
renderer.toneMapping = THREE.ACESFilmicToneMapping with renderer.toneMappingExposure = 1 lets bright lights roll off to white like a camera instead of clipping at pure colour. A THREE.Color takes an sRGB hex (0xff8844) or a CSS string through .setStyle("#ff8844"). Recent three.js writes sRGB output by default, so do not convert the renderer's output a second time.

## three.js lines and wireframes
new THREE.MeshBasicMaterial({ wireframe: true }) draws every triangle edge; a LineSegments with EdgesGeometry draws only the outline of a shape. THREE.Line with LineBasicMaterial and a BufferGeometry of points draws a path, and a line has no thickness on most GPUs, so use TubeGeometry when it must be clearly visible. Line materials are not lit, so their colour is exactly the colour given.

## three.js where Kiwibro has kits
Kiwibro has kits that write a finished three.js page for 3D scenes, 3D objects, particle fields and 3D charts, and a kit for 2D canvas animations; try one of those first and write a three.js page by hand only when none of them fits.
