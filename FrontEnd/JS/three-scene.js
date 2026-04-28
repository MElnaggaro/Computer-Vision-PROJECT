// three-scene.js

// 1. Setup Scene
const container = document.getElementById('three-container');
const scene = new THREE.Scene();

// Camera setup
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.z = 6; // Move camera slightly back to ensure visibility

// Renderer setup
const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
// #three-container already has position: fixed, top: 0, left: 0, pointer-events: none in CSS
container.appendChild(renderer.domElement);

// 2. Lighting (Futuristic AI Style)
const ambientLight = new THREE.AmbientLight(0xffffff, 0.4); // Soft base light
scene.add(ambientLight);

const directionalLight = new THREE.DirectionalLight(0xffffff, 1.2); // Main highlight
directionalLight.position.set(5, 5, 5);
scene.add(directionalLight);

// Subtle colored lights for AI/Futuristic feel
const blueLight = new THREE.PointLight(0x4F46E5, 2, 15); // Accent blue
blueLight.position.set(-3, 3, 3);
scene.add(blueLight);

const purpleLight = new THREE.PointLight(0x9333EA, 2, 15); // Accent purple
purpleLight.position.set(3, -3, 3);
scene.add(purpleLight);

// 3. Model Loading
let model;
let mixer;
const clock = new THREE.Clock();

// Group to hold the model so we can separate base idle animation from scroll transformations
const modelGroup = new THREE.Group();
scene.add(modelGroup);

const loader = new THREE.GLTFLoader();

// Load the 3D model
loader.load(
    'model/aiu 3d.glb', 
    (gltf) => {
        model = gltf.scene;
        
        // Center the model geometry
        const box = new THREE.Box3().setFromObject(model);
        const center = box.getCenter(new THREE.Vector3());
        
        model.position.x = -center.x;
        model.position.y = -center.y;
        model.position.z = -center.z;
        
        // We add the centered model to a pivot group if needed, but adding directly is fine
        // since we use model for idle and modelGroup for scroll.
        modelGroup.add(model);
        
        // Play embedded animations if they exist
        if (gltf.animations && gltf.animations.length > 0) {
            mixer = new THREE.AnimationMixer(model);
            gltf.animations.forEach((clip) => {
                mixer.clipAction(clip).play();
            });
        }

        // Set initial state immediately
        if (states.length > 0) {
            gsap.set(modelGroup.position, states[0].position);
            gsap.set(modelGroup.rotation, states[0].rotation);
        }
    }, 
    undefined, 
    (error) => {
        console.error('An error happened loading the 3D model:', error);
    }
);

// 4. Transform States System for Scroll-Driven Cinematic Experience
const states = [
  {
    id: "hero",
    position: { x: 0, y: 0, z: 0 },
    rotation: { x: 0, y: 0, z: 0 }
  },
  {
    id: "features",
    position: { x: 1.5, y: 0.2, z: 1.0 },
    rotation: { x: 0.2, y: 1.5, z: 0 }
  },
  {
    id: "workflow",
    position: { x: -1.5, y: -0.2, z: 0.5 },
    rotation: { x: 0, y: 3.0, z: 0.1 }
  },
  {
    id: "impact",
    position: { x: 0.5, y: 0.3, z: 0.2 },
    rotation: { x: 0.3, y: 4.5, z: 0 }
  },
  {
    id: "demo",
    position: { x: 0, y: 0, z: 1.5 },
    rotation: { x: 0.1, y: 6.28, z: 0 } // Full rotation to reset visually
  }
];

// 5. Scroll Detection & GSAP Animation
let currentStateId = "hero";

window.addEventListener('scroll', () => {
    if (!modelGroup) return;

    let activeStateId = currentStateId;
    let minDistance = Infinity;
    const centerY = window.innerHeight / 2;

    // Detect which section is active based on distance to center of screen
    states.forEach(state => {
        const element = document.getElementById(state.id);
        if (element) {
            const rect = element.getBoundingClientRect();
            const elementCenterY = rect.top + rect.height / 2;
            const distance = Math.abs(centerY - elementCenterY);
            
            if (distance < minDistance) {
                minDistance = distance;
                activeStateId = state.id;
            }
        }
    });

    // Animate to new state if changed
    if (activeStateId !== currentStateId) {
        currentStateId = activeStateId;
        const targetState = states.find(s => s.id === currentStateId);
        
        if (targetState) {
            // Cinematic transition using GSAP
            gsap.to(modelGroup.position, {
                x: targetState.position.x,
                y: targetState.position.y,
                z: targetState.position.z,
                duration: 1.5, // Smooth duration
                ease: "power3.out",
                overwrite: "auto"
            });
            
            gsap.to(modelGroup.rotation, {
                x: targetState.rotation.x,
                y: targetState.rotation.y,
                z: targetState.rotation.z,
                duration: 1.5,
                ease: "power3.out",
                overwrite: "auto"
            });
        }
    }
});

// 6. Animation Loop (Idle Motion)
const baseIdleTime = { value: 0 };

const animate = () => {
    requestAnimationFrame(animate);
    
    const delta = clock.getDelta();
    baseIdleTime.value += delta;
    
    if (mixer) {
        mixer.update(delta);
    }
    
    if (model) {
        // Base idle continuous rotation (applied to model, not the animating group)
        model.rotation.y += 0.002;
        
        // Base idle floating motion (up/down using sine wave)
        model.position.y += Math.sin(baseIdleTime.value * 2.0) * 0.002;
    }
    
    renderer.render(scene, camera);
};

// Start the animation loop
animate();

// 7. Responsiveness
const handleResize = () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    
    // Adjust model scale for mobile devices to prevent it from dominating the screen
    if (window.innerWidth < 768) {
        modelGroup.scale.set(0.6, 0.6, 0.6);
    } else {
        modelGroup.scale.set(1, 1, 1);
    }
};

window.addEventListener('resize', handleResize);
// Run once on load to set initial scale
handleResize();
