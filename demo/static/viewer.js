import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export var scene,
  canvas,
  canvas_width,
  canvas_height,
  camera,
  renderer,
  points,
  stemBaseMarker,
  markerGroup,
  geometry,
  controls,
  origin_colors;

export async function main() {
  initialize_viewer();
  var data = await loadPointCloud("/pointcloud/current");

  function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
    controls.update();
  }
  animate();
  return data;
}

function initialize_viewer() {
  scene = new THREE.Scene();
  canvas = document.getElementById("viewer");
  canvas_width = canvas.getBoundingClientRect().width;
  canvas_height = canvas.getBoundingClientRect().height;

  camera = new THREE.OrthographicCamera(
    canvas_width / -800,
    canvas_width / 800,
    canvas_height / 800,
    canvas_height / -800,
    1,
    1000
  );
  camera.position.z = 5;

  renderer = new THREE.WebGLRenderer({
    canvas: canvas,
    antialias: true,
  });
  renderer.setSize(canvas_width, canvas_height);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.enableZoom = true;
  controls.enablePan = true;
  controls.screenSpacePanning = true;
  controls.mouseButtons = {
    LEFT: null,
    MIDDLE: THREE.MOUSE.ROTATE,
    RIGHT: THREE.MOUSE.PAN,
  };
  controls.touches = {
    ONE: THREE.TOUCH.ROTATE,
    TWO: THREE.TOUCH.DOLLY_PAN,
  };
}

export async function loadPointCloud(ply_path) {
  var response = await fetch(ply_path);
  var data = await response.json();
  replacePointCloud(data);
  return data;
}

export function replacePointCloud(data) {
  if (points) {
    scene.remove(points);
    geometry.dispose();
  }
  if (markerGroup) {
    scene.remove(markerGroup);
    markerGroup.traverse((obj) => {
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) obj.material.dispose();
    });
    markerGroup = null;
    stemBaseMarker = null;
  }

  var material = new THREE.PointsMaterial({
    size: 6,
    vertexColors: true,
  });

  geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(data.xyz, 3)
  );
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(data.rgb, 3));

  points = new THREE.Points(geometry, material);
  origin_colors = points.geometry.attributes.color.clone();
  scene.add(points);

  const markers = data.base_markers && data.base_markers.length ? data.base_markers : data.stem_base ? [data.stem_base] : [];
  if (markers.length) {
    markerGroup = new THREE.Group();
    var markerGeometry = new THREE.SphereGeometry(0.025, 16, 16);
    for (const marker of markers) {
      var markerMaterial = new THREE.MeshBasicMaterial({ color: 0x111111 });
      const mesh = new THREE.Mesh(markerGeometry.clone(), markerMaterial);
      mesh.position.set(marker[0], marker[1], marker[2]);
      markerGroup.add(mesh);
      if (!stemBaseMarker) stemBaseMarker = mesh;
    }
    scene.add(markerGroup);
  }
}
