import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// Skeleton connections per pose format. The wire protocol is generic, but the
// joint ordering differs between the mock generator and the UmeTrack landmarks
// produced by Meta's emg2pose model.
const CONNECTIONS: Record<string, readonly [number, number][]> = {
  // mock_21: 0 wrist, 1-5 knuckles, 6-10 PIP, 11-15 DIP, 16-20 tips.
  mock_21: [
    [0, 1], [1, 6], [6, 11], [11, 16], // thumb
    [0, 2], [2, 7], [7, 12], [12, 17], // index
    [0, 3], [3, 8], [8, 13], [13, 18], // middle
    [0, 4], [4, 9], [9, 14], [14, 19], // ring
    [0, 5], [5, 10], [10, 15], [15, 20], // pinky
  ],
  // emg2pose_21 (UmeTrack landmarks):
  // 0 thumb_tip, 1 index_tip, 2 middle_tip, 3 ring_tip, 4 pinky_tip,
  // 5 wrist, 6 thumb_intermediate, 7 thumb_distal,
  // 8 index_proximal, 9 index_intermediate, 10 index_distal,
  // 11 middle_proximal, 12 middle_intermediate, 13 middle_distal,
  // 14 ring_proximal, 15 ring_intermediate, 16 ring_distal,
  // 17 pinky_proximal, 18 pinky_intermediate, 19 pinky_distal,
  // 20 palm_center.
  emg2pose_21: [
    [5, 6], [6, 7], [7, 0], // thumb
    [5, 8], [8, 9], [9, 10], [10, 1], // index
    [5, 11], [11, 12], [12, 13], [13, 2], // middle
    [5, 14], [14, 15], [15, 16], [16, 3], // ring
    [5, 17], [17, 18], [18, 19], [19, 4], // pinky
    [5, 20], // palm center to wrist
  ],
};

export interface PoseRenderer {
  readonly format: string;
  updatePose(format: string, joints: readonly (readonly [number, number, number])[], confidence: number): void;
  resize(): void;
  dispose(): void;
}

export function createPoseRenderer(canvas: HTMLCanvasElement): PoseRenderer {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#0b0e14');

  const camera = new THREE.PerspectiveCamera(45, canvas.clientWidth / canvas.clientHeight, 0.01, 100);
  camera.position.set(0, 0.15, 0.25);
  camera.lookAt(0, 0.05, 0);

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'low-power' });
  renderer.setSize(canvas.clientWidth, canvas.clientHeight);
  // Cap pixel ratio so high-DPI displays don't burn GPU on a 4 Hz pose stream.
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  const controls = new OrbitControls(camera, canvas);
  controls.target.set(0, 0.05, 0);
  controls.enableDamping = true;
  controls.dampingFactor = 0.1;

  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const directional = new THREE.DirectionalLight(0xffffff, 1);
  directional.position.set(1, 1, 1);
  scene.add(directional);
  const backLight = new THREE.DirectionalLight(0xffffff, 0.4);
  backLight.position.set(-1, 0.5, -1);
  scene.add(backLight);

  const handGroup = new THREE.Group();
  scene.add(handGroup);

  let currentFormat = 'mock_21';
  let spheres: THREE.Mesh[] = [];
  let boneLines: THREE.Line[] = [];
  let pendingFrame: number | null = null;
  let pageVisible = !document.hidden;

  const jointGeometry = new THREE.SphereGeometry(0.004, 16, 16);
  const jointMaterial = new THREE.MeshStandardMaterial({ color: '#22c55e' });
  const boneMaterial = new THREE.LineBasicMaterial({ color: '#e5e7eb', linewidth: 2 });

  function clearMeshes() {
    for (const child of [...handGroup.children]) {
      handGroup.remove(child);
      if (child instanceof THREE.Mesh || child instanceof THREE.Line) {
        child.geometry.dispose();
      }
    }
    spheres = [];
    boneLines = [];
  }

  function createMeshes(fmt: string) {
    const connections = CONNECTIONS[fmt] ?? CONNECTIONS['mock_21'];
    if (connections === undefined) return;
    clearMeshes();

    for (let i = 0; i < 21; i++) {
      const sphere = new THREE.Mesh(jointGeometry, jointMaterial);
      sphere.visible = false;
      handGroup.add(sphere);
      spheres.push(sphere);
    }

    for (const [start, end] of connections) {
      const geometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0, 0),
        new THREE.Vector3(0, 0, 0),
      ]);
      const line = new THREE.Line(geometry, boneMaterial);
      line.visible = false;
      line.userData = { start, end };
      handGroup.add(line);
      boneLines.push(line);
    }
  }

  createMeshes(currentFormat);

  function render() {
    pendingFrame = null;
    if (!pageVisible) return;
    controls.update();
    renderer.render(scene, camera);
  }

  function scheduleRender() {
    if (pendingFrame === null && pageVisible) {
      pendingFrame = requestAnimationFrame(render);
    }
  }

  function onVisibilityChange() {
    pageVisible = !document.hidden;
    if (pageVisible) {
      scheduleRender();
    } else if (pendingFrame !== null) {
      cancelAnimationFrame(pendingFrame);
      pendingFrame = null;
    }
  }

  document.addEventListener('visibilitychange', onVisibilityChange);
  controls.addEventListener('change', () => {
    scheduleRender();
  });

  function updatePose(fmt: string, joints: readonly [number, number, number][], _confidence: number) {
    if (joints.length !== 21) return;

    if (fmt !== currentFormat) {
      currentFormat = fmt;
      createMeshes(fmt);
    }

    for (let i = 0; i < 21; i++) {
      const [x, y, z] = joints[i] ?? [0, 0, 0];
      const sphere = spheres[i];
      if (sphere !== undefined) {
        sphere.position.set(x, y, z);
        sphere.visible = true;
      }
    }

    for (const line of boneLines) {
      const { start, end } = line.userData as { start: number; end: number };
      const startJoint = joints[start] ?? [0, 0, 0];
      const endJoint = joints[end] ?? [0, 0, 0];
      const positions = line.geometry.attributes['position'] as THREE.BufferAttribute;
      positions.setXYZ(0, startJoint[0], startJoint[1], startJoint[2]);
      positions.setXYZ(1, endJoint[0], endJoint[1], endJoint[2]);
      positions.needsUpdate = true;
      line.geometry.computeBoundingSphere();
      line.visible = true;
    }

    scheduleRender();
  }

  function resize() {
    if (canvas.clientWidth === 0 || canvas.clientHeight === 0) return;
    camera.aspect = canvas.clientWidth / canvas.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(canvas.clientWidth, canvas.clientHeight);
    scheduleRender();
  }

  function dispose() {
    document.removeEventListener('visibilitychange', onVisibilityChange);
    controls.dispose();
    renderer.dispose();
    clearMeshes();
    jointGeometry.dispose();
    jointMaterial.dispose();
    boneMaterial.dispose();
    if (pendingFrame !== null) {
      cancelAnimationFrame(pendingFrame);
    }
  }

  // Initial render so the canvas is not blank.
  scheduleRender();

  return {
    get format() {
      return currentFormat;
    },
    updatePose,
    resize,
    dispose,
  };
}
