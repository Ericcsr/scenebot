
import * as THREE           from 'three';
import { GUI              } from 'three/examples/jsm/libs/lil-gui.module.min.js';
import { OrbitControls    } from 'three/examples/jsm/controls/OrbitControls.js';
import { DragStateManager } from './utils/DragStateManager.js';
import { setupGUI, downloadExampleScenesFolder, loadSceneFromURL, drawTendonsAndFlex, getPosition, getQuaternion, toMujocoPos, standardNormal } from './mujocoUtils.js';
import { PolicyController } from './policy/policyController.js';
import { WSClient, WebKeyboardHandler } from './wsClient.js';
import { awaitStartFlow, captureStartClick } from './spawnClient.js';

// Wire up the Start button click listener immediately at module load. Without
// this, init() takes ~500-1000ms to async-load the scene XML/meshes/policy,
// and a user who clicks Start during that window would have their click lost.
captureStartClick();
import   load_mujoco        from 'mujoco-js/dist/mujoco_wasm.js';
import { loadScenebotAssets, hideDebugGeoms } from './scenebot/loader.js';
import { MotionGraphRuntime, qposFromRuntimePose } from './scenebot/motion_graph_runtime.js';
import { KeyboardCommandState } from './scenebot/keyboard_state.js';
import { PolicyRuntime } from './scenebot/policy_runtime.js';
import {
  createReferenceGhost,
  disposeReferenceGhost,
  setReferenceGhostVisible,
  syncReferenceGhost,
} from './scenebot/reference_motion_viz.js';
import { formatControlsHintPanel, SCENEBOT_PANEL_MIN_WIDTH } from './scenebot/controls_hint.js';

function notifyParentDemo(type, payload = {}) {
  if (window.parent === window) {
    return;
  }
  try {
    window.parent.postMessage({ type, ...payload }, window.location.origin);
  } catch (_) {
    /* ignore */
  }
}

// Load the MuJoCo Module
const mujoco = await load_mujoco();

// Thin-browser web demo: server (run_controller.py + ws_bridge.py) is authoritative.
// Browser loads the SAME merged scene XML the server sims, receives qpos over WS,
// runs FK via mj_forward, and renders via Three.js. No local physics, no policy.
var initialScene = "scene_29dof_flat_hand.xml";
mujoco.FS.mkdir('/working');
mujoco.FS.mount(mujoco.MEMFS, { root: '.' }, '/working');

// Stage the merged scene XML, the included g1 XML, and every <mesh file="..."/>
// referenced by the included XML into Emscripten's MEMFS so MuJoCo's loader can
// resolve relative paths. The `prefix` arg controls where to fetch from:
//   - ws-debug / spawn dev: "/" — files served at site root by run_all.sh /
//     run_spawn.sh, which symlink public/scene_*.xml + public/assets/g1/.
//   - browser (production Pages build): `${BASE_URL}scenebot/` — Vite copied
//     public/scenebot/ flat into dist-desktop/scenebot/, so the iframe at
//     dist-desktop/index.html fetches "./scenebot/scene_*.xml" etc.
async function stageSceneIntoMemfs(rootScene, prefix = "/") {
  const fetchText = async (path) => {
    const r = await fetch(path);
    if (!r.ok) throw new Error(`fetch ${path} failed: ${r.status}`);
    return await r.text();
  };
  const fetchBin = async (path) => {
    const r = await fetch(path);
    if (!r.ok) throw new Error(`fetch ${path} failed: ${r.status}`);
    return new Uint8Array(await r.arrayBuffer());
  };
  const ensureDir = (vfsPath) => {
    const parts = vfsPath.split("/").filter(Boolean);
    let cur = "/working";
    for (let i = 0; i < parts.length - 1; i++) {
      cur += "/" + parts[i];
      if (!mujoco.FS.analyzePath(cur).exists) mujoco.FS.mkdir(cur);
    }
  };

  // 1. Root scene XML.
  const rootXml = await fetchText(prefix + rootScene);
  mujoco.FS.writeFile("/working/" + rootScene, rootXml);

  // 2. Find any <include file="..."/> children referenced by the root scene and stage them.
  const includeRe = /<include\s+file="([^"]+)"\s*\/?>/g;
  const includedFiles = new Set();
  let m;
  while ((m = includeRe.exec(rootXml)) !== null) {
    includedFiles.add(m[1]);
  }
  // 3. For each included XML, fetch + write, and harvest meshdir + <mesh file="..."/>.
  const meshDirByInclude = new Map();
  for (const inc of includedFiles) {
    const incXml = await fetchText(prefix + inc);
    ensureDir(inc);
    mujoco.FS.writeFile("/working/" + inc, incXml);
    const compMatch = incXml.match(/<compiler[^>]*meshdir="([^"]+)"[^>]*\/?>/);
    const meshDir = compMatch ? compMatch[1] : "";
    meshDirByInclude.set(inc, meshDir);
  }
  // 4. For each included XML, walk its <mesh file=".."/> entries and stage each mesh.
  const meshRe = /<mesh\s+[^>]*file="([^"]+)"[^>]*\/?>/g;
  for (const inc of includedFiles) {
    const incXml = mujoco.FS.readFile("/working/" + inc, { encoding: "utf8" });
    const meshDir = meshDirByInclude.get(inc) || "";
    const seen = new Set();
    while ((m = meshRe.exec(incXml)) !== null) {
      const meshFile = m[1];
      if (seen.has(meshFile)) continue;
      seen.add(meshFile);
      // The full path the loader will look up (relative to the cwd at load time, which is /working)
      const relInside = meshDir ? `${meshDir.replace(/\/$/, "")}/${meshFile}` : meshFile;
      const memfsPath = relInside.replace(/^\.\//, "");
      ensureDir(memfsPath);
      const bytes = await fetchBin(prefix + memfsPath);
      mujoco.FS.writeFile("/working/" + memfsPath, bytes);
    }
  }
}

export class MuJoCoDemo {
  constructor() {
    this.mujoco = mujoco;

    // Model and data will be created once assets are available in init()
    this.model = null;
    this.data  = null;

    // Define Random State Variables
    // policyEnabled=false and paused=true: render() short-circuits the local physics + policy
    // pipeline. The WS-driven branch in render() supplies qpos and calls mj_forward instead.
    this.params = { scene: initialScene, paused: true, help: false, ctrlnoiserate: 0.0, ctrlnoisestd: 0.0, keyframeNumber: 0, policyEnabled: false, showRawDepth: false, showReferenceMotion: true };
    this.mujoco_time = 0.0;
    this.bodies  = {}, this.lights = {};
    this.tmpVec  = new THREE.Vector3();
    this.tmpQuat = new THREE.Quaternion();
    this.updateGUICallbacks = [];
    this.policyController = null;
    this.policyStepCounter = 0;
    this.policyDecimation = 1;
    // WS state (populated in init()).
    this.ws = null;
    this.webKeys = null;
    this.boxQposAdr = -1;
    this.pelvisFollowOffset = new THREE.Vector3(-4.0, 1.5, 0.0);
    this.defaultJointPos = [
      0.162997201, -0.0361181423, -0.0214254409, 0.267154634, -0.174296871, 0.212671682,
      0.282425106, -0.0584460497, -0.556104779, 0.126711249, -0.123827517, -0.190653816,
      0.000492588617, -0.0195334535, 0.428676069,
      -0.00628881808, 0.161155701, 0.236345276, 0.980316162, 0.15456377, 0.0774896815, 0.0205286704, -0.128641531,
      -0.0847690701, -0.255017966, 1.09530210, -0.134532213, 0.0875737667, 0.0601755157
    ];
    this.container = document.createElement( 'div' );
    document.body.appendChild( this.container );
    const guiMode = import.meta.env.VITE_GUI_MODE || 'open';
    const showMobileButtons = guiMode === 'hide';

    this.speedControlsContainer = document.createElement('div');
    this.speedControlsContainer.style.position = 'absolute';
    if (showMobileButtons) {
      this.speedControlsContainer.style.top = '16px';
      this.speedControlsContainer.style.right = '16px';
    } else {
      this.speedControlsContainer.style.top = '64px';
      this.speedControlsContainer.style.left = '16px';
    }
    this.speedControlsContainer.style.display = 'flex';
    this.speedControlsContainer.style.alignItems = 'center';
    this.speedControlsContainer.style.gap = '10px';
    this.speedControlsContainer.style.zIndex = '1200';

    this.speedModeElement = document.createElement('div');
    this.speedModeElement.style.padding = '8px 12px';
    this.speedModeElement.style.borderRadius = '8px';
    this.speedModeElement.style.background = 'rgba(0, 0, 0, 0.60)';
    this.speedModeElement.style.color = '#ffffff';
    this.speedModeElement.style.font = 'bold 16px Arial';
    this.speedModeElement.style.letterSpacing = '0.2px';

    this.speedControlsContainer.appendChild(this.speedModeElement);
    if (showMobileButtons) {
      this.speedToggleButton = document.createElement('button');
      this.speedToggleButton.type = 'button';
      this.speedToggleButton.textContent = 'Toggle speed';
      this.speedToggleButton.style.padding = '8px 12px';
      this.speedToggleButton.style.borderRadius = '8px';
      this.speedToggleButton.style.border = 'none';
      this.speedToggleButton.style.background = 'rgba(0, 0, 0, 0.60)';
      this.speedToggleButton.style.color = '#ffffff';
      this.speedToggleButton.style.font = 'bold 14px Arial';
      this.speedToggleButton.style.letterSpacing = '0.2px';
      this.speedToggleButton.style.cursor = 'pointer';
      this.speedToggleButton.addEventListener('click', () => {
        if (this.policyController) {
          this.policyController.highSpeedMode = !this.policyController.highSpeedMode;
          this.updateSpeedModeIndicator();
        }
      });

      this.resetButton = document.createElement('button');
      this.resetButton.type = 'button';
      this.resetButton.textContent = 'Reset';
      this.resetButton.style.padding = '8px 12px';
      this.resetButton.style.borderRadius = '8px';
      this.resetButton.style.border = 'none';
      this.resetButton.style.background = 'rgba(0, 0, 0, 0.60)';
      this.resetButton.style.color = '#ffffff';
      this.resetButton.style.font = 'bold 14px Arial';
      this.resetButton.style.letterSpacing = '0.2px';
      this.resetButton.style.cursor = 'pointer';
      this.resetButton.addEventListener('click', () => {
        if (typeof this.reloadScene === 'function') {
          this.reloadScene();
        }
      });

      this.speedControlsContainer.appendChild(this.speedToggleButton);
      this.speedControlsContainer.appendChild(this.resetButton);

      this.refMotionButton = document.createElement('button');
      this.refMotionButton.type = 'button';
      this.refMotionButton.textContent = 'Ref motion';
      this.refMotionButton.style.padding = '8px 12px';
      this.refMotionButton.style.borderRadius = '8px';
      this.refMotionButton.style.border = 'none';
      this.refMotionButton.style.background = 'rgba(0, 0, 0, 0.60)';
      this.refMotionButton.style.color = '#ffffff';
      this.refMotionButton.style.font = 'bold 14px Arial';
      this.refMotionButton.style.letterSpacing = '0.2px';
      this.refMotionButton.style.cursor = 'pointer';
      this.refMotionButton.addEventListener('click', () => {
        this.params.showReferenceMotion = !this.params.showReferenceMotion;
        setReferenceGhostVisible(this.refViz, this.params.showReferenceMotion);
        this.refMotionButton.style.opacity = this.params.showReferenceMotion ? '1' : '0.45';
      });
      this.refMotionButton.style.display = 'none';
      this.speedControlsContainer.appendChild(this.refMotionButton);
    }
    window.addEventListener('keydown', (event) => {
      if (event.key === 'Backspace') {
        if (typeof this.reloadScene === 'function') {
          this.reloadScene();
        }
        event.preventDefault();
      }
    });
    document.body.appendChild(this.speedControlsContainer);
    this.updateSpeedModeIndicator();

    this.scene = new THREE.Scene();
    this.scene.name = 'scene';

    this.camera = new THREE.PerspectiveCamera( 45, window.innerWidth / window.innerHeight, 0.001, 500 );
    this.camera.name = 'PerspectiveCamera';
    this.camera.position.set(2.0, 1.7, 1.7);
    this.scene.add(this.camera);

    // Secondary camera for depth inset view.
    // Match d435i depth config in far-tracking.
    this.depthCameraConfig = {
      width: 106,
      height: 60,
      horizontalFovDeg: 58.4,
      minRange: 0.3,
      maxRange: 3.0,
    };
    this.depthCameraView = new THREE.PerspectiveCamera(
      this.depthCameraConfig.horizontalFovDeg,
      this.depthCameraConfig.width / this.depthCameraConfig.height,
      this.depthCameraConfig.minRange,
      this.depthCameraConfig.maxRange
    );
    this.depthCameraView.position.set(3.0, 2.0, 3.0);
    this.depthCameraView.lookAt(0, 0.7, 0);
    this.depthCameraView.layers.set(1);
    this.scene.add(this.depthCameraView);
    this.depthCameraPoseViz = new THREE.Group();
    this.depthCameraPoseViz.name = 'DepthCameraPoseViz';
    this.depthCameraMarker = new THREE.Mesh(
      new THREE.SphereGeometry(0.02, 20, 20),
      new THREE.MeshBasicMaterial({ color: 0xff4d9d })
    );
    this.depthCameraPoseViz.add(this.depthCameraMarker);
    this.depthCameraView.add(this.depthCameraPoseViz);
    // Scenebot demo doesn't use the depth camera; hide its pose marker so it doesn't
    // show up as a giant red sphere in the main view (layer-isolation isn't reliable
    // when the marker is parented under a Camera that's added to the main scene).
    this.depthCameraPoseViz.visible = false;

    this.scene.background = new THREE.Color(0.15, 0.25, 0.35);
    // Fog: (color, near, far). Increase far so distant terrain stays visible.
    // this.scene.fog = new THREE.Fog(this.scene.background, 30, 120);

    this.ambientLight = new THREE.AmbientLight( 0xffffff, 0.1 * 3.14 );
    this.ambientLight.name = 'AmbientLight';
    this.scene.add( this.ambientLight );

    this.spotlight = new THREE.SpotLight();
    this.spotlight.angle = 1.11;
    this.spotlight.distance = 10000;
    this.spotlight.penumbra = 0.5;
    this.spotlight.castShadow = true; // default false
    this.spotlight.intensity = this.spotlight.intensity * 3.14 * 10.0;
    this.spotlight.shadow.mapSize.width = 1024; // default
    this.spotlight.shadow.mapSize.height = 1024; // default
    this.spotlight.shadow.camera.near = 0.1; // default
    this.spotlight.shadow.camera.far = 100; // default
    this.spotlight.position.set(0, 3, 3);
    const targetObject = new THREE.Object3D();
    this.scene.add(targetObject);
    this.spotlight.target = targetObject;
    targetObject.position.set(0, 1, 0);
    this.scene.add( this.spotlight );

    // Extra fill lights for clearer scene visibility.
    this.hemiLight = new THREE.HemisphereLight(0xbfd8ff, 0x1f2a3a, 0.35 * 2.0);
    this.hemiLight.position.set(0, 6, 0);
    this.scene.add(this.hemiLight);

    this.fillLightLeft = new THREE.DirectionalLight(0xffffff, 0.28 * 2.0);
    this.fillLightLeft.position.set(-4, 3, 2);
    this.scene.add(this.fillLightLeft);

    this.fillLightRight = new THREE.DirectionalLight(0xffffff, 0.22 * 2.0);
    this.fillLightRight.position.set(4, 2.5, -1.5);
    this.scene.add(this.fillLightRight);

    this.renderer = new THREE.WebGLRenderer( { antialias: true } );
    this.renderer.setPixelRatio(1.0);////window.devicePixelRatio );
    this.renderer.setSize( window.innerWidth, window.innerHeight );
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap; // default THREE.PCFShadowMap
    THREE.ColorManagement.enabled = false;
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    //this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    //this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    //this.renderer.toneMappingExposure = 2.0;
    this.renderer.useLegacyLights = true;

    this.container.appendChild( this.renderer.domElement );

    // Depth render target and material for visualization.
    // Keep render resolution fixed to the camera config.
    const depthPreviewScale = Number(import.meta.env.VITE_DEPTH_PREVIEW_SCALE ?? 4);
    this.depthInset = {
      width: this.depthCameraConfig.width,
      height: this.depthCameraConfig.height,
      margin: 16,
      previewScale: depthPreviewScale,
    };
    this.depthCameraView.aspect = this.depthCameraConfig.width / this.depthCameraConfig.height;
    this.depthCameraView.updateProjectionMatrix();
    this.depthTarget = new THREE.WebGLRenderTarget(
      this.depthInset.width,
      this.depthInset.height
    );
    this.depthTarget.texture.minFilter = THREE.NearestFilter;
    this.depthTarget.texture.magFilter = THREE.NearestFilter;
    this.depthTarget.texture.generateMipmaps = false;
    this.depthTarget.depthTexture = new THREE.DepthTexture();
    this.depthTarget.depthTexture.format = THREE.DepthFormat;
    this.depthTarget.depthTexture.type = THREE.FloatType;
    this.depthViewMaterial = new THREE.ShaderMaterial({
      uniforms: {
        tDepth: { value: this.depthTarget.depthTexture },
        cameraNear: { value: this.depthCameraView.near },
        cameraFar: { value: this.depthCameraView.far },
        depthScale: { value: 10.0 }, // clamp distance to improve contrast
      },
      vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = vec4(position.xy, 0.0, 1.0);
        }
      `,
      fragmentShader: `
        #include <packing>
        uniform sampler2D tDepth;
        uniform float cameraNear;
        uniform float cameraFar;
        uniform float depthScale;
        varying vec2 vUv;
        void main() {
          float depth = texture2D(tDepth, vUv).x;
          float viewZ = perspectiveDepthToViewZ(depth, cameraNear, cameraFar);
          float linearDepth = -viewZ;
          float v = clamp(linearDepth / depthScale, 0.0, 1.0);
          gl_FragColor = vec4(vec3(v), 1.0);
        }
      `,
    });
    this.depthRawScene = new THREE.Scene();
    this.depthCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.depthRawPixels = new Uint8Array(this.depthCameraConfig.width * this.depthCameraConfig.height * 4);
    this.depthRawTexture = new THREE.DataTexture(
      this.depthRawPixels,
      this.depthCameraConfig.width,
      this.depthCameraConfig.height,
      THREE.RGBAFormat
    );
    this.depthRawTexture.minFilter = THREE.NearestFilter;
    this.depthRawTexture.magFilter = THREE.NearestFilter;
    this.depthRawTexture.needsUpdate = true;
    this.depthRawMaterial = new THREE.MeshBasicMaterial({ map: this.depthRawTexture });
    this.depthRawMesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.depthRawMaterial);
    this.depthRawScene.add(this.depthRawMesh);
    this.depthPreviewSize = { width: 87, height: 58 };
    this.depthProcessedInset = {
      width: this.depthPreviewSize.width,
      height: this.depthPreviewSize.height,
      gap: 8,
      scale: depthPreviewScale,
    };
    this.depthPreviewPixels = new Uint8Array(this.depthPreviewSize.width * this.depthPreviewSize.height * 4);
    this.depthPreviewTexture = new THREE.DataTexture(
      this.depthPreviewPixels,
      this.depthPreviewSize.width,
      this.depthPreviewSize.height,
      THREE.RGBAFormat
    );
    this.depthPreviewTexture.minFilter = THREE.NearestFilter;
    this.depthPreviewTexture.magFilter = THREE.NearestFilter;
    this.depthPreviewTexture.needsUpdate = true;
    this.depthPreviewMaterial = new THREE.MeshBasicMaterial({ map: this.depthPreviewTexture });
    this.depthPreviewMesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.depthPreviewMaterial);
    this.depthProcessedScene = new THREE.Scene();
    this.depthProcessedScene.add(this.depthPreviewMesh);

    this.depthInferenceMaterial = new THREE.ShaderMaterial({
      uniforms: {
        tDepth: { value: this.depthTarget.depthTexture },
        cameraNear: { value: this.camera.near },
        cameraFar: { value: this.camera.far }
      },
      vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = vec4(position.xy, 0.0, 1.0);
        }
      `,
      fragmentShader: `
        #include <packing>
        uniform sampler2D tDepth;
        uniform float cameraNear;
        uniform float cameraFar;
        varying vec2 vUv;
        void main() {
          float depth = texture2D(tDepth, vUv).x;
          float viewZ = perspectiveDepthToViewZ(depth, cameraNear, cameraFar);
          float linearDepth = -viewZ;
          gl_FragColor = vec4(linearDepth, 0.0, 0.0, 1.0);
        }
      `,
    });
    this.depthInferenceScene = new THREE.Scene();
    this.depthInferenceScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.depthInferenceMaterial));
    this.depthInferenceTarget = new THREE.WebGLRenderTarget(
      this.depthInset.width,
      this.depthInset.height,
      {
        type: THREE.FloatType,
        format: THREE.RGBAFormat,
        minFilter: THREE.NearestFilter,
        magFilter: THREE.NearestFilter,
        depthBuffer: false,
        stencilBuffer: false,
      }
    );
    this.depthPixels = new Float32Array(this.depthInset.width * this.depthInset.height * 4);
    this.depthFrame = new Float32Array(this.depthInset.width * this.depthInset.height);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0.7, 0);
    this.controls.panSpeed = 2;
    this.controls.zoomSpeed = 1;
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.10;
    this.controls.screenSpacePanning = true;
    this.controls.update();

    window.addEventListener('resize', this.onWindowResize.bind(this));

    // Initialize the Drag State Manager.
    this.dragStateManager = new DragStateManager(this.scene, this.renderer, this.camera, this.container.parentElement, this.controls);
  }

  async init() {
    // Route selection driven by <body data-mode>:
    //   data-mode="spawn"      → POST /sessions, wait for ready, connect to per-user ws
    //                            (this is what index.html / the public demo serves)
    //   data-mode="ws-debug"   → connect to a fixed shared-sim ws (run_all.sh debug)
    //   absent / "browser"     → full-browser: sim + policy + motion graph in JS
    //
    // ?backend=ws://host:port still works: it overrides the WS URL when
    // data-mode="ws-debug", and is also a back-compat shortcut to force WS mode.
    const params = new URLSearchParams(window.location.search);
    const backendUrl = params.get("backend");
    const bodyMode = (document.body && document.body.dataset.mode) || "";
    if (bodyMode === "spawn") {
      this.runMode = "spawn";
    } else if (bodyMode === "ws-debug" || backendUrl) {
      this.runMode = "ws";
    } else {
      this.runMode = "browser";
    }
    this.backendUrl = backendUrl || "ws://" + location.hostname + ":8765";
    console.log(`[scenebot] runMode=${this.runMode}` + (this.runMode === "ws" ? ` backend=${this.backendUrl}` : ""));

    // Stage scene XML + meshes into MEMFS. Where to fetch them depends on
    // runMode (see stageSceneIntoMemfs comment for the rationale):
    //   browser → import.meta.env.BASE_URL + "scenebot/" so the production
    //             build at dist-desktop/index.html can fetch "./scenebot/...".
    //   ws / spawn → "/" because the launchers symlink the assets at site root.
    const stagePrefix = this.runMode === "browser"
      ? `${import.meta.env.BASE_URL}scenebot/`.replace(/\/+/g, "/")
      : "/";
    await stageSceneIntoMemfs(initialScene, stagePrefix);

    // Download the the examples to MuJoCo's virtual file system
    await downloadExampleScenesFolder(mujoco);

    // Initialize the three.js Scene using the .xml Model in initialScene
    [this.model, this.data, this.bodies, this.lights] =
      await loadSceneFromURL(mujoco, initialScene, this);

    this.applySceneInitialState({ resetData: false, rebindCameras: true });

    // mujoco_wasm's scene loader inserts orphan debug geoms (a red cylinder +
    // sphere at world origin) for any geom it can't attach to a body. They show
    // up as a giant red blob in the middle of the scene. Hide them in every
    // mode (was previously only called from _initFullBrowser).
    hideDebugGeoms(this.scene);

    // Resolve the qpos address of the dynamic free box (if it exists in the scene).
    this._resolveBoxQposAdr();

    if (this.runMode === "spawn") {
      // Wait for the user to click Start. captureStartClick() at module-load
      // time already wired the listener, so a click during init's async
      // bootstrapping is latched.
      const overlay = document.getElementById("startOverlay");
      const { wsUrl } = await awaitStartFlow();
      console.log("[scene] spawn-mode connecting to", wsUrl);
      this.ws = new WSClient(wsUrl);
      this.webKeys = new WebKeyboardHandler(this.ws);
      // Hide the overlay once the first WS frame arrives, not before — that
      // way users don't see a blank scene if the WS handshake fails.
      const hideOnFrame = () => {
        if (this.ws && this.ws.latestQpos && overlay) {
          overlay.style.display = "none";
          return true;
        }
        return false;
      };
      const overlayHider = setInterval(() => {
        if (hideOnFrame()) clearInterval(overlayHider);
      }, 100);
    } else if (this.runMode === "ws") {
      console.log("[scene] connecting to", this.backendUrl);
      this.ws = new WSClient(this.backendUrl);
      this.webKeys = new WebKeyboardHandler(this.ws);
    } else {
      await this._initFullBrowser();
    }

    this.gui = new GUI();
    setupGUI(this);

    // Start the render loop only after the model and assets are ready
    this.renderer.setAnimationLoop( this.render.bind(this) );
  }

  async _initFullBrowser() {
    // Load packaged scenebot assets (motion graph, clips, contact labels, policy meta).
    const assets = await loadScenebotAssets({
      onProgress: (m) => {
        console.log("[load]", m);
        notifyParentDemo("scenebot-demo-progress", { message: m });
      },
    });

    this.motionGraph = new MotionGraphRuntime(
      assets.motionGraph,
      assets.clipBundle,
      assets.contactLabels,
      {
        fps: assets.policyMeta.control_dt > 0 ? 1.0 / assets.policyMeta.control_dt : 50.0,
        streamContactDim: assets.contactLabels.maxStreamDim(),
        defaultContactLabel: assets.policyMeta.default_contact_label,
        contactLabelsMnOnly: true,
        pickupForwardStepScale: 0.5, // matches --slow-pickup-2x default
      },
    );

    this.kb = new KeyboardCommandState();
    this.kb.attachDom();

    this.policy = await PolicyRuntime.create(assets.policyOnnxUrl, assets.policyMeta);

    // Persistent reusable scratch arrays.
    this._lastTargetQ = null;
    this._policyDt = assets.policyMeta.control_dt; // 0.02 s = 50 Hz
    this._simDt = assets.policyMeta.simulation_dt; // 0.005 s = 200 Hz
    this._stepsPerControl = Math.max(1, Math.round(this._policyDt / this._simDt));
    // Crucially align mj_step's integrator dt with the Python config. mujoco_env.py
    // line 357 sets `model.opt.timestep = config.simulation_dt` (0.005). The scene
    // XML doesn't carry a <option timestep=...>, so without this line we'd inherit
    // MuJoCo's 0.002 default and the browser robot would run in slower sim-time.
    this.model.opt.timestep = this._simDt;
    this._kp = Float32Array.from(assets.policyMeta.joint_stiffness);
    this._kd = Float32Array.from(assets.policyMeta.joint_damping);
    this._torqueLimit = Float32Array.from(assets.policyMeta.torque_limit);
    this._yawRateRadPerS = (60 * Math.PI) / 180; // mirrors --yaw-adjust-deg-per-s default
    this._initQpos36 = Array.isArray(assets.policyMeta.init_qpos_36)
      ? assets.policyMeta.init_qpos_36
      : null;

    this._applyFullBrowserInitialPose();

    this._policyAccumMs = 0;
    this._lastTickerMs = -1;

    // Perf overlay: right-side panel showing live policy/physics/render rates and timing.
    this._perf = {
      tickCount: 0,
      renderCount: 0,
      lastResetMs: performance.now(),
      // ring buffers (last N timings, in ms)
      stepDurMs: new Float32Array(120),
      stepIdx: 0,
      inferDurMs: new Float32Array(120),
      inferIdx: 0,
      mjStepDurMs: new Float32Array(120),
      mjStepIdx: 0,
    };
    this._perfDom = this._installPerfOverlay();
    this.speedModeElement.style.display = 'none';
    const guiModeBrowser = import.meta.env.VITE_GUI_MODE || 'open';
    if (guiModeBrowser === 'open') {
      this.speedControlsContainer.style.display = 'none';
    }
    if (guiModeBrowser === 'hide') {
      this._installKeyboardHintInSpeedPanel();
    }
    this._setupReferenceGhost();
    this._seedReferencePoseFromSim();
    if (this.refMotionButton) {
      this.refMotionButton.style.display = "";
      this.refMotionButton.style.opacity = this.params.showReferenceMotion ? "1" : "0.45";
    }
    console.log("[scenebot] full-browser runtime initialized");
    notifyParentDemo("scenebot-demo-ready");

    // Drive the sim with a setTimeout loop so cadence is independent of rAF
    // throttling (Chromium throttles rAF heavily when the page is occluded /
    // backgrounded / under headless Xvfb).
    this._startSimLoop();
  }

  _applyFullBrowserInitialPose() {
    if (!this.model || !this.data) return;
    this.mujoco.mj_resetData(this.model, this.data);
    // Initialize sim qpos[0:36] from policyMeta.init_qpos_36 (frame 0 of the ref
    // motion, baked offline by build_browser_assets.py to match Python's
    // mujoco_env.py:213-219). This puts the robot at the same world location and
    // joint configuration the Python controller starts with — necessary for
    // end-to-end physics parity with the WS-bridge backend.
    if (this._initQpos36 && this._initQpos36.length === 36) {
      const q0 = this._initQpos36;
      for (let i = 0; i < 36 && i < this.data.qpos.length; i++) this.data.qpos[i] = q0[i];
      this.mujoco.mj_forward(this.model, this.data);
    } else if (this.motionGraph && this.policy) {
      // Fallback: fall back to MotionGraphRuntime's stop frame. Less accurate but
      // keeps the demo runnable when init_qpos_36 isn't baked into policy_meta.
      console.warn("[scene] policy_meta.init_qpos_36 missing; falling back to motion graph Stop pose");
      const init = this.motionGraph.step("Stop", null, 0);
      const q0 = qposFromRuntimePose(
        init.joint_pos_isaac, init.root_pos_w, init.root_quat_wxyz, this.policy.ISAAC_TO_MUJOCO,
      );
      for (let i = 0; i < 36 && i < this.data.qpos.length; i++) this.data.qpos[i] = q0[i];
      this.mujoco.mj_forward(this.model, this.data);
    }
  }

  async onSceneReloaded() {
    if (this.runMode !== "browser") return;
    this.model.opt.timestep = this._simDt;
    this._resolveBoxQposAdr();
    if (this.motionGraph) this.motionGraph.reset();
    this._applyFullBrowserInitialPose();
    if (this.kb) this.kb.reset();
    if (this.policy) this.policy.reset();
    this._policyAccumMs = 0;
    this._lastTickerMs = -1;
    this.bindTrackingTargets();
    this._setupReferenceGhost();
    this._seedReferencePoseFromSim();
  }

  _disposeReferenceGhost() {
    disposeReferenceGhost(this.refViz);
    this.refViz = null;
    this._latestRefQpos = null;
  }

  _setupReferenceGhost() {
    if (this.runMode !== "browser" || !this.model || !this.bodies) return;
    this._disposeReferenceGhost();
    this.refViz = createReferenceGhost({
      scene: this.scene,
      mujoco: this.mujoco,
      model: this.model,
      bodies: this.bodies,
    });
    setReferenceGhostVisible(this.refViz, this.params.showReferenceMotion !== false);
  }

  _seedReferencePoseFromSim() {
    if (!this.data?.qpos || !this.policy) return;
    this._latestRefQpos = new Float64Array(36);
    for (let i = 0; i < 36; i++) this._latestRefQpos[i] = this.data.qpos[i];
  }

  _updateLatestRefQpos(packet) {
    if (!packet || !this.policy || !this.kb) return;
    const jpRef = this.kb.blendJointPoseUpperFrozen(packet.joint_pos_isaac);
    this._latestRefQpos = qposFromRuntimePose(
      jpRef,
      packet.root_pos_w,
      packet.root_quat_wxyz,
      this.policy.ISAAC_TO_MUJOCO,
    );
  }

  _syncReferenceMotion() {
    if (!this.params.showReferenceMotion || !this.refViz || !this._latestRefQpos) return;
    const rootZ = this.data?.qpos?.length > 2 ? this.data.qpos[2] : undefined;
    syncReferenceGhost(this.mujoco, this.model, this.refViz, this._latestRefQpos, { rootZ });
  }

  _resolveBoxQposAdr() {
    this.boxQposAdr = -1;
    try {
      const MJ_OBJ_BODY = 1;
      const boxBodyId = mujoco.mj_name2id(this.model, MJ_OBJ_BODY, "free_box");
      if (boxBodyId >= 0) {
        const jntAdr = this.model.body_jntadr[boxBodyId];
        if (jntAdr >= 0) {
          this.boxQposAdr = this.model.jnt_qposadr[jntAdr];
        }
      }
    } catch (err) {
      console.warn("[scene] could not resolve free_box qpos addr:", err);
    }
  }

  _installPerfOverlay() {
    const wrap = document.createElement("div");
    wrap.id = "scenebot-perf-wrap";
    wrap.style.cssText = [
      "position:absolute",
      "top:108px",
      "left:16px",
      "z-index:1300",
      "display:flex",
      "flex-direction:column",
      "gap:8px",
      "pointer-events:none",
      `min-width:${SCENEBOT_PANEL_MIN_WIDTH}`,
      "width:max-content",
    ].join(";");

    const root = document.createElement("div");
    root.id = "scenebot-perf";
    root.style.cssText = [
      "padding:10px 14px",
      "background:rgba(0,0,0,0.65)",
      "color:#cfe9ff",
      "font:12px/1.4 ui-monospace,Menlo,Consolas,monospace",
      "border-radius:8px",
      "white-space:pre",
      `min-width:${SCENEBOT_PANEL_MIN_WIDTH}`,
      "box-sizing:border-box",
    ].join(";");
    root.textContent = "perf: warming up…";

    wrap.appendChild(root);
    (this.container || document.body).appendChild(wrap);
    return root;
  }

  _installKeyboardHintInSpeedPanel() {
    if (this._keyboardHintDom || !this.speedControlsContainer) return;

    const topRow = document.createElement("div");
    topRow.style.cssText = "display:flex;align-items:center;gap:10px;flex-wrap:wrap";
    while (this.speedControlsContainer.firstChild) {
      topRow.appendChild(this.speedControlsContainer.firstChild);
    }

    const hint = document.createElement("div");
    hint.id = "scenebot-controls-hint-speed";
    hint.style.cssText = [
      "padding:6px 10px",
      "border-radius:8px",
      "background:rgba(0,0,0,0.55)",
      "color:#dfefff",
      "font:11px/1.35 Arial,sans-serif",
      `min-width:${SCENEBOT_PANEL_MIN_WIDTH}`,
      "width:max-content",
      "max-width:min(520px, calc(100vw - 32px))",
      "white-space:pre-wrap",
      "box-sizing:border-box",
    ].join(";");
    hint.textContent = formatControlsHintPanel();

    const guiMode = import.meta.env.VITE_GUI_MODE || "open";
    this.speedControlsContainer.style.flexDirection = "column";
    this.speedControlsContainer.style.alignItems = guiMode === "hide" ? "flex-end" : "flex-start";
    this.speedControlsContainer.style.minWidth = SCENEBOT_PANEL_MIN_WIDTH;
    this.speedControlsContainer.style.maxWidth = "min(520px, calc(100vw - 32px))";
    this.speedControlsContainer.appendChild(topRow);
    this.speedControlsContainer.appendChild(hint);
    this._keyboardHintDom = hint;
  }

  _updatePerfOverlay() {
    if (!this._perfDom) return;
    const now = performance.now();
    const dt = now - this._perf.lastResetMs;
    if (dt < 500) return;
    const policyHz = (this._perf.tickCount * 1000) / dt;
    const renderHz = (this._perf.renderCount * 1000) / dt;
    const avg = (arr) => {
      let s = 0, n = 0;
      for (let i = 0; i < arr.length; i++) if (arr[i] > 0) { s += arr[i]; n++; }
      return n ? s / n : 0;
    };
    const max = (arr) => {
      let m = 0;
      for (let i = 0; i < arr.length; i++) if (arr[i] > m) m = arr[i];
      return m;
    };
    const stepAvg = avg(this._perf.stepDurMs);
    const stepMax = max(this._perf.stepDurMs);
    const infAvg = avg(this._perf.inferDurMs);
    const infMax = max(this._perf.inferDurMs);
    const mjAvg = avg(this._perf.mjStepDurMs);
    const mjMax = max(this._perf.mjStepDurMs);
    const targetPolicyHz = 1.0 / this._policyDt;
    const ratio = (policyHz / targetPolicyHz) * 100;

    const line = (label, val, unit) =>
      `${label.padEnd(11)} ${String(val).padStart(7)} ${unit}`;
    const fmt = (n, d = 1) => Number.isFinite(n) ? n.toFixed(d) : "—";

    this._perfDom.textContent = [
      `policy   ${fmt(policyHz, 1).padStart(5)} Hz   ` +
        `target ${targetPolicyHz.toFixed(0)} Hz  (${fmt(ratio, 0)}%)`,
      `render   ${fmt(renderHz, 1).padStart(5)} Hz`,
      `step     avg ${fmt(stepAvg, 2).padStart(6)} ms   max ${fmt(stepMax, 2)} ms`,
      `  ONNX     ${fmt(infAvg, 2).padStart(6)} ms   max ${fmt(infMax, 2)} ms`,
      `  mj_step×${this._stepsPerControl} ${fmt(mjAvg, 2).padStart(6)} ms   max ${fmt(mjMax, 2)} ms`,
    ].join("\n");

    // reset per-second window
    this._perf.tickCount = 0;
    this._perf.renderCount = 0;
    this._perf.lastResetMs = now;
  }

  bindTrackingTargets() {
    this.pelvisBody = Object.values(this.bodies).find(
      (body) => body && body.name === 'pelvis'
    );
    if (this.pelvisBody) {
      const pelvisPos = this.pelvisBody.position.clone();
      this.camera.position.copy(pelvisPos).add(this.pelvisFollowOffset);
      this.controls.target.copy(pelvisPos);
      this.controls.update();
    }

    const depthAnchorCandidates = [
      'torso_link',
      'torso',
      'trunk',
      'waist_roll_link',
      'pelvis',
    ];
    this.depthCameraAnchorBody = depthAnchorCandidates
      .map((name) => Object.values(this.bodies).find((body) => body && body.name === name))
      .find((body) => !!body) || null;
    if (!this.depthCameraAnchorBody) {
      this.depthCameraAnchorBody =
        Object.values(this.bodies).find((body) => body && typeof body.name === 'string' && body.name.includes('torso')) ||
        this.pelvisBody ||
        null;
    }

    if (this.depthCameraAnchorBody) {
      this.depthCameraAnchorBody.add(this.depthCameraView);
      console.log('Depth camera anchor body:', this.depthCameraAnchorBody.name);

      const offsetPos = { x: 0.01, y: 0.01, z: 0.44 };
      this.depthCameraView.position.set(
        offsetPos.x,
        offsetPos.z,
        -offsetPos.y,
      );

      const deg2rad = THREE.MathUtils.degToRad;
      const xAxis = new THREE.Vector3(1, 0, 0);
      const yAxis = new THREE.Vector3(0, 1, 0);
      const zAxis = new THREE.Vector3(0, 0, 1);
      const rpyDegToMjQuat = (rollDeg, pitchDeg, yawDeg) => {
        const qx = new THREE.Quaternion().setFromAxisAngle(xAxis, deg2rad(rollDeg));
        const qy = new THREE.Quaternion().setFromAxisAngle(yAxis, deg2rad(pitchDeg));
        const qz = new THREE.Quaternion().setFromAxisAngle(zAxis, deg2rad(yawDeg));
        return qz.multiply(qy).multiply(qx);
      };
      const mjQuatToThreeQuat = (qMj) =>
        new THREE.Quaternion(-qMj.x, -qMj.z, qMj.y, -qMj.w);

      const qOffsetMj = rpyDegToMjQuat(1, 27, 1);
      const qBaseMj = rpyDegToMjQuat(0, 0, -90);
      const qSensorMj = qOffsetMj.multiply(qBaseMj);
      this.depthCameraView.quaternion.copy(mjQuatToThreeQuat(qSensorMj).normalize());
    } else {
      console.warn('Depth camera anchor body not found; using world-fixed camera.');
    }
  }

  updateSpeedModeIndicator() {
    if (!this.speedModeElement || this.runMode === 'browser') {
      return;
    }
    const isHighSpeed = this.policyController ? this.policyController.highSpeedMode !== false : true;
    this.speedModeElement.textContent = `Speed: ${isHighSpeed ? 'HIGH' : 'LOW'}`;
  }

  applySceneInitialState({ resetData = false, rebindCameras = false } = {}) {
    if (!this.model || !this.data) {
      return;
    }
    if (resetData) {
      this.mujoco.mj_resetData(this.model, this.data);
    }

    const isPrimaryDemoScene = this.params.scene === initialScene;
    const startQpos = 7;
    const startQvel = 6;
    const canApplyJointInit =
      isPrimaryDemoScene &&
      (startQpos + this.defaultJointPos.length) <= this.data.qpos.length &&
      (startQvel + this.defaultJointPos.length) <= this.data.qvel.length;
    if (canApplyJointInit) {
      for (let i = 0; i < this.defaultJointPos.length; i++) {
        this.data.qpos[startQpos + i] = this.defaultJointPos[i];
      }
      for (let i = 0; i < this.defaultJointPos.length; i++) {
        this.data.qvel[startQvel + i] = 0.0;
      }
    }

    this.mujoco.mj_forward(this.model, this.data);
    if (rebindCameras) {
      this.bindTrackingTargets();
    }
    if (this.policyController && typeof this.policyController.reset === 'function') {
      this.policyController.reset();
    }
    this.policyStepCounter = 0;
  }

  async initPolicy() {
    const urlParams = new URLSearchParams(window.location.search);
    const defaultPolicyPath = './2026-01-17_09-51-30_student-new-loco-old-skill_student.onnx';
    const modelPath = urlParams.get('policy') || defaultPolicyPath;
    const controller = new PolicyController(this.mujoco, {
      modelPath: modelPath,
      depthModelPath: urlParams.get('depthPolicy') || modelPath.replace('_student.onnx', '_depth_backbone.onnx'),
      controlDt: 0.02
    });
    try {
      await controller.init(this.model);
      this.policyController = controller;
      this.policyStepCounter = 0;
      const timestep = this.model?.opt?.timestep ?? 0.002;
      this.policyDecimation = Math.max(1, Math.round(controller.controlDt / timestep));
      console.log('Policy loaded. Decimation:', this.policyDecimation);
    } catch (error) {
      console.error('Failed to initialize policy:', error);
      this.policyController = null;
    }
  }

  async rebuildPolicy() {
    if (!this.policyController) {
      return;
    }
    try {
      await this.policyController.rebuild(this.model);
      this.policyStepCounter = 0;
    } catch (error) {
      console.error('Failed to rebuild policy:', error);
    }
  }

  onWindowResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize( window.innerWidth, window.innerHeight );
    this.depthInset.width = this.depthCameraConfig.width;
    this.depthInset.height = this.depthCameraConfig.height;
    this.depthTarget.setSize(this.depthCameraConfig.width, this.depthCameraConfig.height);
    this.depthInferenceTarget.setSize(this.depthCameraConfig.width, this.depthCameraConfig.height);
    this.depthCameraView.aspect = this.depthCameraConfig.width / this.depthCameraConfig.height;
    this.depthCameraView.updateProjectionMatrix();
    this.depthPixels = new Float32Array(this.depthCameraConfig.width * this.depthCameraConfig.height * 4);
    this.depthFrame = new Float32Array(this.depthCameraConfig.width * this.depthCameraConfig.height);
  }

  /**
   * Full-browser per-RAF tick: drives the motion graph + policy at 50 Hz and steps physics
   * at 200 Hz, mirroring the cadence of the Python backend (control_dt=0.02, simulation_dt=0.005).
   *
   * @param {number} timeMS  performance.now() from setAnimationLoop
   */
  async _tickFullBrowser(timeMS) {
    if (this._lastTickerMs < 0) {
      this._lastTickerMs = timeMS;
      this._policyAccumMs = 0;
    }
    let elapsedMs = timeMS - this._lastTickerMs;
    if (elapsedMs > 250) elapsedMs = 250;
    this._lastTickerMs = timeMS;
    this._policyAccumMs += elapsedMs;
    if (this._perf) this._perf.renderCount++;

    const policyDtMs = this._policyDt * 1000;
    while (this._policyAccumMs >= policyDtMs) {
      this._policyAccumMs -= policyDtMs;
      const t0 = performance.now();
      await this._stepFullBrowserOnce();
      if (this._perf) {
        const dur = performance.now() - t0;
        this._perf.stepDurMs[this._perf.stepIdx] = dur;
        this._perf.stepIdx = (this._perf.stepIdx + 1) % this._perf.stepDurMs.length;
        this._perf.tickCount++;
      }
    }
    this._updatePerfOverlay();
  }

  async _stepFullBrowserOnce() {
    // 1) Pull keyboard state.
    const desired = this.kb.getCommand();
    const ctrlSkipCmd = this.kb.pollCtrlSkipToStop();
    const sitToggleCmd = this.kb.pollSitToggleCommand();
    const latchedCommand = sitToggleCmd != null ? sitToggleCmd : desired;
    const dyaw = this.kb.pollYawAdjustment(this._policyDt, this._yawRateRadPerS);

    // 2) Step motion graph → stream packet (lower_cmd / vr_* / contact_mask / motion_anchor_*).
    const packet = this.motionGraph.step(latchedCommand, ctrlSkipCmd, dyaw);

    // Auto-freeze upper body when forward pick-up (G) finishes — snapshot the final pose.
    if (packet.pickup_forward_completed && !this.kb.upperBodyFreezeEnabled()) {
      this.kb.activateUpperBodyFreeze();
      this.kb.setUpperBodyFreezeSnapshot(
        packet.joint_pos_isaac, packet.contact_mask, packet.vr_3point_pos_l, packet.vr_3point_orn_l,
      );
    }

    // 3) Optional upper-body freeze (F): take snapshot when freeze toggles enabled.
    const freezeEv = this.kb.pollUpperBodyFreezeToggle();
    if (freezeEv === "enabled") {
      this.kb.setUpperBodyFreezeSnapshot(
        packet.joint_pos_isaac, packet.contact_mask, packet.vr_3point_pos_l, packet.vr_3point_orn_l,
      );
    } else if (freezeEv === "disabled") {
      this.kb.clearUpperBodyFreezeSnapshot();
    }
    this.kb.applyFrozenUpperToStreamPacket(packet);
    this._updateLatestRefQpos(packet);

    // 4) Feed packet into the policy's "latest streaming inputs".
    this.policy.ingestStreamPacket(packet);

    // 5) Build robotState snapshot from MuJoCo.
    const robotState = this._readRobotState();

    // 6) Run policy.
    const controlSignals = this.policy.prepareControlSignals(robotState);
    const obs = this.policy.prepareObs(robotState, controlSignals);
    const tInf = performance.now();
    const rawAction = await this.policy.getAction(obs);
    if (this._perf) {
      const dur = performance.now() - tInf;
      this._perf.inferDurMs[this._perf.inferIdx] = dur;
      this._perf.inferIdx = (this._perf.inferIdx + 1) % this._perf.inferDurMs.length;
    }
    const targetQ = this.policy.applyControl(rawAction);
    this._lastTargetQ = targetQ;

    // 7) Inner physics loop: step MuJoCo `_stepsPerControl` times with PD control toward targetQ.
    const tMj = performance.now();
    this._innerPhysicsSteps(targetQ);
    if (this._perf) {
      const dur = performance.now() - tMj;
      this._perf.mjStepDurMs[this._perf.mjStepIdx] = dur;
      this._perf.mjStepIdx = (this._perf.mjStepIdx + 1) % this._perf.mjStepDurMs.length;
    }
  }

  _readRobotState() {
    // Pull qpos[7:36] (29 joints), qvel[6:35] (29 vel), qpos[3:7] (root quat wxyz), qvel[3:6] (omega),
    // qpos[0:3] (root pos), qvel[0:3] (root vel) out of MuJoCo. Match what mujoco_env.py shares.
    const qpos = this.data.qpos, qvel = this.data.qvel;
    const q = new Float32Array(29);
    const dq = new Float32Array(29);
    for (let i = 0; i < 29; i++) { q[i] = qpos[7 + i]; dq[i] = qvel[6 + i]; }
    const omega = new Float32Array([qvel[3], qvel[4], qvel[5]]);
    // imu_quat is wxyz (G1RobotState comment); root_orn is xyzw (mocap convention).
    const wxyz = [qpos[3], qpos[4], qpos[5], qpos[6]];
    const imu_quat = Float32Array.from(wxyz);
    const root_orn = Float32Array.from([wxyz[1], wxyz[2], wxyz[3], wxyz[0]]);
    const root_pos = Float32Array.from([qpos[0], qpos[1], qpos[2]]);
    const root_vel = Float32Array.from([qvel[0], qvel[1], qvel[2]]);
    return { q, dq, omega, imu_quat, root_pos, root_orn, root_vel };
  }

  _innerPhysicsSteps(targetQ) {
    // PD: data.ctrl = (targetQ - q) * kp - dq * kd, clipped to torque limits.
    const ctrl = this.data.ctrl;
    const qpos = this.data.qpos, qvel = this.data.qvel;
    for (let s = 0; s < this._stepsPerControl; s++) {
      for (let i = 0; i < 29; i++) {
        let tau = (targetQ[i] - qpos[7 + i]) * this._kp[i] - qvel[6 + i] * this._kd[i];
        const lim = this._torqueLimit[i];
        if (tau > lim) tau = lim;
        else if (tau < -lim) tau = -lim;
        ctrl[i] = tau;
      }
      mujoco.mj_step(this.model, this.data);
    }
  }

  /** Pull the latest data.xpos/xquat into Three.js body transforms. Cheap
   *  enough to call every rAF frame; safe to call without holding a sim
   *  step lock because we only read MuJoCo arrays. */
  _syncBodyTransformsFromMujoco() {
    for (let bIdx = 0; bIdx < this.model.nbody; bIdx++) {
      if (this.bodies[bIdx]) {
        getPosition(this.data.xpos, bIdx, this.bodies[bIdx].position);
        getQuaternion(this.data.xquat, bIdx, this.bodies[bIdx].quaternion);
        this.bodies[bIdx].updateWorldMatrix();
      }
    }
    if (this.pelvisBody && this.pelvisFollowOffset) {
      this.camera.position.copy(this.pelvisBody.position).add(this.pelvisFollowOffset);
      this.controls.target.copy(this.pelvisBody.position);
      this.controls.update();
    }
    for (let l = 0; l < this.model.nlight; l++) {
      if (this.lights[l]) {
        getPosition(this.data.light_xpos, l, this.lights[l].position);
        getPosition(this.data.light_xdir, l, this.tmpVec);
        this.lights[l].lookAt(this.tmpVec.add(this.lights[l].position));
      }
    }
  }

  /** setTimeout-driven sim loop: runs the motion graph + policy + physics at
   *  a real 50 Hz wall-clock cadence, decoupled from rAF so it stays fast even
   *  when the renderer is throttled (occluded tab, headless, etc.). Each tick
   *  re-arms a setTimeout for the next slot, accounting for the time the
   *  previous tick took. */
  _startSimLoop() {
    if (this._simLoopRunning) return;
    this._simLoopRunning = true;
    const policyDtMs = this._policyDt * 1000;
    let nextDeadline = performance.now() + policyDtMs;
    const tick = async () => {
      if (!this._simLoopRunning) return;
      const tBefore = performance.now();
      try {
        await this._stepFullBrowserOnce();
        if (this._perf) {
          const dur = performance.now() - tBefore;
          this._perf.stepDurMs[this._perf.stepIdx] = dur;
          this._perf.stepIdx = (this._perf.stepIdx + 1) % this._perf.stepDurMs.length;
          this._perf.tickCount++;
        }
      } catch (e) {
        console.error("[sim loop] step failed:", e);
      }
      // Self-correcting cadence: figure out when next tick should fire.
      nextDeadline += policyDtMs;
      let delay = nextDeadline - performance.now();
      // If we fell more than 100ms behind (background tab woke up), drop
      // accumulated debt rather than firing a burst.
      if (delay < -100) {
        nextDeadline = performance.now() + policyDtMs;
        delay = policyDtMs;
      }
      if (delay < 0) delay = 0;
      setTimeout(tick, delay);
    };
    setTimeout(tick, policyDtMs);
  }

  _stopSimLoop() {
    this._simLoopRunning = false;
  }

  async render(timeMS) {
    // If the model isn't ready yet, skip rendering this frame.
    if (!this.model || !this.data) {
      return;
    }
    this.updateSpeedModeIndicator();
    this.controls.update();

    // Full-browser mode: motion graph + policy + sim run on a separate
    // setTimeout-driven loop (started in _initFullBrowser). The render path
    // here only syncs Three.js body transforms and draws — keeping it light
    // means sim cadence stays at 50 Hz even when rAF is throttled (background
    // tab, headless Chromium, occluded window, etc.).
    if (this.runMode === "browser" && this.motionGraph && this.policy) {
      this._syncBodyTransformsFromMujoco();
      this._syncReferenceMotion();
      this.renderer.render(this.scene, this.camera);
      return;
    }

    // Thin-browser web demo: server is authoritative. Pull the latest qpos from the WS
    // client, write into MuJoCo's data buffer, run FK, then fall through to the body
    // transform sync below. This bypasses the local physics loop and policy entirely.
    if (this.ws && this.ws.latestQpos) {
      const q = this.ws.latestQpos; // Float32Array(36)
      // qpos layout: [root_xyz(3), root_quat_wxyz(4), joints(29), ...].
      // The first 36 entries map 1:1 to data.qpos[0:36] for the G1 model.
      for (let i = 0; i < 36 && i < this.data.qpos.length; i++) {
        this.data.qpos[i] = q[i];
      }
      if (this.boxQposAdr >= 0 && this.ws.latestBox) {
        const b = this.ws.latestBox; // Float32Array(7)
        for (let i = 0; i < 7 && (this.boxQposAdr + i) < this.data.qpos.length; i++) {
          this.data.qpos[this.boxQposAdr + i] = b[i];
        }
      }
      mujoco.mj_forward(this.model, this.data);

      // Update body / light transforms (mirrors the post-step block lower in the
      // function; keeping it here so we can early-return).
      for (let bIdx = 0; bIdx < this.model.nbody; bIdx++) {
        if (this.bodies[bIdx]) {
          getPosition  (this.data.xpos , bIdx, this.bodies[bIdx].position);
          getQuaternion(this.data.xquat, bIdx, this.bodies[bIdx].quaternion);
          this.bodies[bIdx].updateWorldMatrix();
        }
      }
      if (this.pelvisBody && this.pelvisFollowOffset) {
        this.camera.position.copy(this.pelvisBody.position).add(this.pelvisFollowOffset);
        this.controls.target.copy(this.pelvisBody.position);
        this.controls.update();
      }
      for (let l = 0; l < this.model.nlight; l++) {
        if (this.lights[l]) {
          getPosition(this.data.light_xpos, l, this.lights[l].position);
          getPosition(this.data.light_xdir, l, this.tmpVec);
          this.lights[l].lookAt(this.tmpVec.add(this.lights[l].position));
        }
      }

      this.renderer.render(this.scene, this.camera);
      return;
    }

    // Auto-forward when robot is near terrain boxes (climbing zones)
    if (this.policyController && this.params.policyEnabled && this.params.scene === initialScene) {
      const pelvisX = this.data.qpos[0];
      const boxXPositions = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60];
      const nearBox = boxXPositions.some(bx => pelvisX >= bx - 1.5 && pelvisX <= bx + 1.0);
      this.policyController.autoForward = nearBox;
      this.policyController._updateCommandState();
    }

    if (!this.params["paused"]) {
      let timestep = this.model.opt.timestep;
      if (timeMS - this.mujoco_time > 35.0) { this.mujoco_time = timeMS; }
      while (this.mujoco_time < timeMS) {

        // Jitter the control state with gaussian random noise
        if (this.params["ctrlnoisestd"] > 0.0) {
          let rate  = Math.exp(-timestep / Math.max(1e-10, this.params["ctrlnoiserate"]));
          let scale = this.params["ctrlnoisestd"] * Math.sqrt(1 - rate * rate);
          let currentCtrl = this.data.ctrl;
          for (let i = 0; i < currentCtrl.length; i++) {
            currentCtrl[i] = rate * currentCtrl[i] + scale * standardNormal();
            this.params["Actuator " + i] = currentCtrl[i];
          }
        }

        // Clear old perturbations, apply new ones.
        for (let i = 0; i < this.data.qfrc_applied.length; i++) { this.data.qfrc_applied[i] = 0.0; }
        let dragged = this.dragStateManager.physicsObject;
        if (dragged && dragged.bodyID) {
          for (let b = 0; b < this.model.nbody; b++) {
            if (this.bodies[b]) {
              getPosition  (this.data.xpos , b, this.bodies[b].position);
              getQuaternion(this.data.xquat, b, this.bodies[b].quaternion);
              this.bodies[b].updateWorldMatrix();
            }
          }
          let bodyID = dragged.bodyID;
          this.dragStateManager.update(); // Update the world-space force origin
          let force = toMujocoPos(this.dragStateManager.currentWorld.clone().sub(this.dragStateManager.worldHit).multiplyScalar(this.model.body_mass[bodyID] * 250));
          let point = toMujocoPos(this.dragStateManager.worldHit.clone());
          mujoco.mj_applyFT(this.model, this.data, [force.x, force.y, force.z], [0, 0, 0], [point.x, point.y, point.z], bodyID, this.data.qfrc_applied);

          // TODO: Apply pose perturbations (mocap bodies only).
        }

        if (this.policyController && this.params.policyEnabled) {
          if (this.policyStepCounter % this.policyDecimation === 0) {
            try {
              await this.policyController.requestAction(this.model, this.data);
            } catch (error) {
              console.error('Policy inference error:', error);
            }
          }
          this.policyController.applyControl(this.model, this.data);
          this.policyStepCounter += 1;
        }

        mujoco.mj_step(this.model, this.data);

        this.mujoco_time += timestep * 1000.0;
      }

    } else if (this.params["paused"]) {
      this.dragStateManager.update(); // Update the world-space force origin
      let dragged = this.dragStateManager.physicsObject;
      if (dragged && dragged.bodyID) {
        let b = dragged.bodyID;
        getPosition  (this.data.xpos , b, this.tmpVec , false); // Get raw coordinate from MuJoCo
        getQuaternion(this.data.xquat, b, this.tmpQuat, false); // Get raw coordinate from MuJoCo

        let offset = toMujocoPos(this.dragStateManager.currentWorld.clone()
          .sub(this.dragStateManager.worldHit).multiplyScalar(0.3));
        if (this.model.body_mocapid[b] >= 0) {
          // Set the root body's mocap position...
          console.log("Trying to move mocap body", b);
          let addr = this.model.body_mocapid[b] * 3;
          let pos  = this.data.mocap_pos;
          pos[addr+0] += offset.x;
          pos[addr+1] += offset.y;
          pos[addr+2] += offset.z;
        } else {
          // Set the root body's position directly...
          let root = this.model.body_rootid[b];
          let addr = this.model.jnt_qposadr[this.model.body_jntadr[root]];
          let pos  = this.data.qpos;
          pos[addr+0] += offset.x;
          pos[addr+1] += offset.y;
          pos[addr+2] += offset.z;
        }
      }

      mujoco.mj_forward(this.model, this.data);
    }

    // Update body transforms.
    for (let b = 0; b < this.model.nbody; b++) {
      if (this.bodies[b]) {
        getPosition  (this.data.xpos , b, this.bodies[b].position);
        getQuaternion(this.data.xquat, b, this.bodies[b].quaternion);
        this.bodies[b].updateWorldMatrix();
      }
    }

    if (this.pelvisBody && this.pelvisFollowOffset) {
      this.camera.position.copy(this.pelvisBody.position).add(this.pelvisFollowOffset);
      this.controls.target.copy(this.pelvisBody.position);
      this.controls.update();
    }
    // Update light transforms.
    for (let l = 0; l < this.model.nlight; l++) {
      if (this.lights[l]) {
        getPosition(this.data.light_xpos, l, this.lights[l].position);
        getPosition(this.data.light_xdir, l, this.tmpVec);
        this.lights[l].lookAt(this.tmpVec.add(this.lights[l].position));
      }
    }

    // Draw Tendons and Flex verts
    drawTendonsAndFlex(this.mujocoRoot, this.model, this.data);
    this.depthViewMaterial.uniforms.cameraNear.value = this.depthCameraView.near;
    this.depthViewMaterial.uniforms.cameraFar.value = this.depthCameraView.far;

    // Render main view to screen.
    this.renderer.setRenderTarget(null);
    this.renderer.setViewport(0, 0, window.innerWidth, window.innerHeight);
    this.renderer.setScissorTest(false);
    this.renderer.clear();
    this.renderer.render(this.scene, this.camera);

    // Render depth from the secondary camera into a target.
    this.renderer.setRenderTarget(this.depthTarget);
    this.renderer.clear();
    this.renderer.render(this.scene, this.depthCameraView);
    this.renderer.setRenderTarget(null);

    // Render depth into a float target for inference (linear depth in meters).
    this.depthInferenceMaterial.uniforms.cameraNear.value = this.depthCameraView.near;
    this.depthInferenceMaterial.uniforms.cameraFar.value = this.depthCameraView.far;
    this.renderer.setRenderTarget(this.depthInferenceTarget);
    this.renderer.clear();
    this.renderer.render(this.depthInferenceScene, this.depthCamera);
    this.renderer.readRenderTargetPixels(
      this.depthInferenceTarget,
      0,
      0,
      this.depthInset.width,
      this.depthInset.height,
      this.depthPixels
    );
    this.renderer.setRenderTarget(null);
    const showRawDepth = !!this.params.showRawDepth;

    if (this.policyController) {
      const width = this.depthInset.width;
      const height = this.depthInset.height;
      const pixelCount = width * height;
      for (let i = 0; i < pixelCount; i++) {
        this.depthFrame[i] = this.depthPixels[i * 4];
      }
      this.policyController.setDepthImage(this.depthFrame, width, height);
      if (this.depthRawPixels && this.depthRawTexture && showRawDepth) {
        const minDepth = 0.3;
        const maxDepth = 3.0;
        const range = maxDepth - minDepth;
        for (let i = 0; i < this.depthFrame.length; i++) {
          const v = Number.isFinite(this.depthFrame[i]) ? this.depthFrame[i] : maxDepth;
          const norm = Math.max(0.0, Math.min(1.0, (v - minDepth) / range));
          const c = Math.round(norm * 255);
          const base = i * 4;
          this.depthRawPixels[base] = c;
          this.depthRawPixels[base + 1] = c;
          this.depthRawPixels[base + 2] = c;
          this.depthRawPixels[base + 3] = 255;
        }
        this.depthRawTexture.needsUpdate = true;
      }
    }

    if (this.policyController?.getProcessedDepthPreview) {
      const preview = this.policyController.getProcessedDepthPreview();
      if (preview) {
        const { data, width, height } = preview;
        if (width !== this.depthPreviewSize.width || height !== this.depthPreviewSize.height) {
          this.depthPreviewSize = { width, height };
          this.depthProcessedInset.width = width;
          this.depthProcessedInset.height = height;
          this.depthPreviewPixels = new Uint8Array(width * height * 4);
          this.depthPreviewTexture.dispose();
          this.depthPreviewTexture = new THREE.DataTexture(
            this.depthPreviewPixels,
            width,
            height,
            THREE.RGBAFormat
          );
          this.depthPreviewTexture.minFilter = THREE.NearestFilter;
          this.depthPreviewTexture.magFilter = THREE.NearestFilter;
          this.depthPreviewMaterial.map = this.depthPreviewTexture;
          this.depthPreviewMaterial.needsUpdate = true;
        }
        const pixelCount = width * height;
        const minDepth = 0.3;
        const maxDepth = 3.0;
        const range = maxDepth - minDepth;     
  
        for (let i = 0; i < pixelCount; i++) {
          // const v = Math.max(0, Math.min(1, data[i] + 0.5));

          let v = Number.isFinite(data[i]) ? (data[i] + 0.5) : 0.0;
          v = Math.max(0.0, Math.min(1.0, v));
          const c = Math.round(v * 255);
          const base = i * 4;
          this.depthPreviewPixels[base] = c;
          this.depthPreviewPixels[base + 1] = c;
          this.depthPreviewPixels[base + 2] = c;
          this.depthPreviewPixels[base + 3] = 255;
        }
        this.depthPreviewTexture.needsUpdate = true;
      }
    }

    // Visualize raw + processed depth in small inset viewports.
    this.renderer.setScissorTest(true);
    const rawX = this.depthInset.margin;
    const rawY = this.depthInset.margin;
    const rawW = this.depthInset.width * this.depthInset.previewScale;
    const rawH = this.depthInset.height * this.depthInset.previewScale;
    if (showRawDepth) {
      this.renderer.setViewport(rawX, rawY, rawW, rawH);
      this.renderer.setScissor(rawX, rawY, rawW, rawH);
      this.renderer.render(this.depthRawScene, this.depthCamera);
    }

    const processedX = showRawDepth ? (rawX + rawW + this.depthProcessedInset.gap) : rawX;
    const processedY = rawY;
    const processedW = this.depthProcessedInset.width * this.depthProcessedInset.scale;
    const processedH = this.depthProcessedInset.height * this.depthProcessedInset.scale;
    this.renderer.setViewport(processedX, processedY, processedW, processedH);
    this.renderer.setScissor(processedX, processedY, processedW, processedH);
    this.renderer.render(this.depthProcessedScene, this.depthCamera);
    this.renderer.setScissorTest(false);
    this.renderer.setViewport(0, 0, window.innerWidth, window.innerHeight);
  }
}

let demo = new MuJoCoDemo();
window.demo = demo; // exposed for headless verification + browser console debugging
await demo.init();
