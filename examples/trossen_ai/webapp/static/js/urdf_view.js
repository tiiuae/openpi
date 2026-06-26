// Trimmed single-panel URDF viewer for the Replay preview. Loads /robot.urdf,
// drives 14-D joints, orbit + view presets. Adapted from the wizard viewer.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { ColladaLoader } from 'three/examples/jsm/loaders/ColladaLoader.js';
import URDFLoader from 'urdf-loader';

// Dataset 14-D index -> URDF joint name (left 0-6, right 7-13).
const JOINT_MAP = {
  0: 'follower_left_joint_0',  1: 'follower_left_joint_1',  2: 'follower_left_joint_2',
  3: 'follower_left_joint_3',  4: 'follower_left_joint_4',  5: 'follower_left_joint_5',
  7: 'follower_right_joint_0', 8: 'follower_right_joint_1', 9: 'follower_right_joint_2',
  10: 'follower_right_joint_3', 11: 'follower_right_joint_4', 12: 'follower_right_joint_5',
};
const GRIPPER_MAP = {
  6:  ['follower_left_right_carriage_joint',  'follower_left_left_carriage_joint'],
  13: ['follower_right_right_carriage_joint', 'follower_right_left_carriage_joint'],
};

const VIEWS = {
  top:   { pos: [0, 0, 5],      up: [0, 1, 0] },
  front: { pos: [0, -4.5, 1.2], up: [0, 0, 1] },
  left:  { pos: [-4.5, 0, 1.2], up: [0, 0, 1] },
  right: { pos: [4.5, 0, 1.2],  up: [0, 0, 1] },
  back:  { pos: [0, 4.5, 1.2],  up: [0, 0, 1] },
  iso:   { pos: [3.0, -3.5, 2.0], up: [0, 0, 1] },
};

export class UrdfView {
  constructor(canvas) {
    this.canvas = canvas;
    this.robot = null;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#f0f2f5');
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.8));
    const sun = new THREE.DirectionalLight(0xffffff, 1.2);
    sun.position.set(3, 2, 5);
    this.scene.add(sun);
    const grid = new THREE.GridHelper(6, 30, 0x999999, 0xcccccc);
    grid.rotation.x = Math.PI / 2;
    this.scene.add(grid);

    this.camera = new THREE.PerspectiveCamera(45, this._aspect(), 0.01, 50);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(...VIEWS.iso.pos);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.set(0, 0, 1);
    this.controls.enableDamping = true;
    this.controls.update();

    window.addEventListener('resize', () => this._resize());
    this._resize();
    this._animate();
  }

  _aspect() {
    const p = this.canvas.parentElement;
    return Math.max(1, p.clientWidth) / Math.max(1, p.clientHeight);
  }

  _resize() {
    const p = this.canvas.parentElement;
    this.renderer.setSize(p.clientWidth, p.clientHeight, false);
    this.camera.aspect = this._aspect();
    this.camera.updateProjectionMatrix();
  }

  _animate() {
    requestAnimationFrame(() => this._animate());
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  _makeLoader() {
    const loader = new URDFLoader();
    loader.packages = { trossen_arm_description: '/pkg/trossen_arm_description' };
    loader.loadMeshCb = (path, manager, done) => {
      if (/\.dae$/i.test(path)) {
        new ColladaLoader(manager).load(path, dae => {
          dae.scene.updateMatrixWorld(true);
          done(dae.scene);
        }, null, err => done(null, err));
      } else {
        loader.defaultMeshLoader(path, manager, done);
      }
    };
    return loader;
  }

  async load() {
    const loader = this._makeLoader();
    this.robot = await new Promise((resolve, reject) => {
      loader.load('/robot.urdf', r => {
        r.traverse(c => {
          if (c.isMesh) {
            c.material = new THREE.MeshPhongMaterial({ color: 0x8899aa, shininess: 60 });
          }
        });
        resolve(r);
      }, null, reject);
    });
    this.scene.add(this.robot);
  }

  // Apply a 14-D joint vector (radians + gripper metres) to the URDF.
  setFrameJoints(joints14) {
    if (!this.robot || !joints14) return;
    for (const [idx, name] of Object.entries(JOINT_MAP)) {
      const v = joints14[+idx];
      if (v !== undefined) this.robot.setJointValue(name, v);
    }
    for (const [idx, names] of Object.entries(GRIPPER_MAP)) {
      const v = joints14[+idx];
      if (v !== undefined) for (const n of names) this.robot.setJointValue(n, v);
    }
  }

  snapView(name) {
    const v = VIEWS[name] || VIEWS.iso;
    this.camera.up.set(...v.up);
    this.camera.position.set(...v.pos);
    this.controls.target.set(0, 0, 1);
    this.controls.update();
  }
}
