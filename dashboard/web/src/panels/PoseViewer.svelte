<script lang="ts">
  import { onMount } from 'svelte';
  import { on } from '../lib/socket.svelte';
  import type { PoseFrame } from '../lib/protocol';
  import * as THREE from 'three';

  // Joint layout for the mock_21 format produced by the Python pose service.
  // 0: wrist
  // 1-5: MCP knuckles (thumb, index, middle, ring, pinky)
  // 6-10: PIP joints
  // 11-15: DIP joints
  // 16-20: fingertips
  const JOINT_CONNECTIONS: readonly [number, number][] = [
    // thumb
    [0, 1], [1, 6], [6, 11], [11, 16],
    // index
    [0, 2], [2, 7], [7, 12], [12, 17],
    // middle
    [0, 3], [3, 8], [8, 13], [13, 18],
    // ring
    [0, 4], [4, 9], [9, 14], [14, 19],
    // pinky
    [0, 5], [5, 10], [10, 15], [15, 20],
  ];

  let canvas: HTMLCanvasElement | undefined = $state(undefined);
  let confidence = $state(0);
  let format = $state('');

  let scene: THREE.Scene | null = null;
  let camera: THREE.PerspectiveCamera | null = null;
  let renderer: THREE.WebGLRenderer | null = null;
  let handGroup: THREE.Group | null = null;
  let spheres: THREE.Mesh[] = [];
  let boneLines: THREE.Line[] = [];

  function initRenderer() {
    if (canvas === undefined) return;
    scene = new THREE.Scene();
    scene.background = new THREE.Color('#0b0e14');

    camera = new THREE.PerspectiveCamera(45, canvas.clientWidth / canvas.clientHeight, 0.01, 100);
    camera.position.set(0, 0.15, 0.25);
    camera.lookAt(0, 0.05, 0);

    renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setSize(canvas.clientWidth, canvas.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const directional = new THREE.DirectionalLight(0xffffff, 1);
    directional.position.set(1, 1, 1);
    scene.add(directional);

    handGroup = new THREE.Group();
    scene.add(handGroup);

    createHandMeshes();
  }

  function createHandMeshes() {
    if (handGroup === null) return;
    const jointMaterial = new THREE.MeshStandardMaterial({ color: '#22c55e' });
    const jointGeometry = new THREE.SphereGeometry(0.004, 16, 16);
    const boneMaterial = new THREE.LineBasicMaterial({ color: '#e5e7eb', linewidth: 2 });

    spheres = [];
    for (let i = 0; i < 21; i++) {
      const sphere = new THREE.Mesh(jointGeometry, jointMaterial);
      sphere.visible = false;
      handGroup.add(sphere);
      spheres.push(sphere);
    }

    boneLines = [];
    for (const [start, end] of JOINT_CONNECTIONS) {
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

  function updatePose(pose: PoseFrame) {
    if (pose.joints.length !== 21) return;

    confidence = pose.confidence;
    format = pose.format;

    if (scene === null) initRenderer();
    if (scene === null || camera === null || renderer === null) return;

    for (let i = 0; i < 21; i++) {
      const [x, y, z] = pose.joints[i] ?? [0, 0, 0];
      const sphere = spheres[i];
      if (sphere !== undefined) {
        sphere.position.set(x, y, z);
        sphere.visible = true;
      }
    }

    for (const line of boneLines) {
      const { start, end } = line.userData as { start: number; end: number };
      const startJoint = pose.joints[start] ?? [0, 0, 0];
      const endJoint = pose.joints[end] ?? [0, 0, 0];
      const positions = line.geometry.attributes['position'] as THREE.BufferAttribute;
      positions.setXYZ(0, startJoint[0], startJoint[1], startJoint[2]);
      positions.setXYZ(1, endJoint[0], endJoint[1], endJoint[2]);
      positions.needsUpdate = true;
      line.geometry.computeBoundingSphere();
      line.visible = true;
    }

    renderer.render(scene, camera);
  }

  function handleResize() {
    if (canvas === undefined || camera === null || renderer === null) return;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (width === 0 || height === 0) return;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
    renderer.render(scene!, camera);
  }

  onMount(() => {
    initRenderer();
    const offPose = on('pose', updatePose);
    window.addEventListener('resize', handleResize);
    return () => {
      offPose();
      window.removeEventListener('resize', handleResize);
      renderer?.dispose();
    };
  });
</script>

<h2>Pose</h2>
<div class="row">
  <span class="muted">{format ? `${format} · ` : ''}confidence {confidence.toFixed(2)}</span>
</div>

<div class="canvas-wrap">
  <canvas bind:this={canvas}></canvas>
</div>

<style>
  .canvas-wrap {
    width: 100%;
    height: 60vh;
    min-height: 320px;
  }
  canvas {
    width: 100%;
    height: 100%;
    display: block;
  }
</style>
