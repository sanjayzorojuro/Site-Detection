/**
 * Three.js 3D Scene for the Homepage Hero Section
 * Creates animated wireframe construction geometry with particles
 */
(function () {
    const canvas = document.getElementById('hero-canvas');
    if (!canvas) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 2, 8);

    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0);

    // ── Wireframe Construction Crane ──
    const craneMat = new THREE.LineBasicMaterial({ color: 0x00e5ff, transparent: true, opacity: 0.25 });

    // Vertical tower
    const towerGeo = new THREE.BoxGeometry(0.6, 6, 0.6);
    const towerEdges = new THREE.EdgesGeometry(towerGeo);
    const tower = new THREE.LineSegments(towerEdges, craneMat);
    tower.position.set(-2, 0, 0);
    scene.add(tower);

    // Horizontal boom
    const boomGeo = new THREE.BoxGeometry(7, 0.3, 0.3);
    const boomEdges = new THREE.EdgesGeometry(boomGeo);
    const boom = new THREE.LineSegments(boomEdges, craneMat);
    boom.position.set(1.5, 3, 0);
    scene.add(boom);

    // Building wireframe
    const buildGeo = new THREE.BoxGeometry(2.5, 3, 2.5);
    const buildEdges = new THREE.EdgesGeometry(buildGeo);
    const buildMat = new THREE.LineBasicMaterial({ color: 0x7c4dff, transparent: true, opacity: 0.15 });
    const building = new THREE.LineSegments(buildEdges, buildMat);
    building.position.set(3, -1.5, -2);
    scene.add(building);

    // Small structure
    const smallGeo = new THREE.BoxGeometry(1.5, 1.8, 1.5);
    const smallEdges = new THREE.EdgesGeometry(smallGeo);
    const smallMat = new THREE.LineBasicMaterial({ color: 0x00e5ff, transparent: true, opacity: 0.12 });
    const smallBuild = new THREE.LineSegments(smallEdges, smallMat);
    smallBuild.position.set(-3.5, -2.1, -1);
    scene.add(smallBuild);

    // ── Floating Particles ──
    const particleCount = 300;
    const positions = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount; i++) {
        positions[i * 3] = (Math.random() - 0.5) * 20;
        positions[i * 3 + 1] = (Math.random() - 0.5) * 14;
        positions[i * 3 + 2] = (Math.random() - 0.5) * 14;
    }
    const particleGeo = new THREE.BufferGeometry();
    particleGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const particleMat = new THREE.PointsMaterial({
        color: 0x00e5ff, size: 0.04, transparent: true, opacity: 0.5,
        blending: THREE.AdditiveBlending, depthWrite: false,
    });
    const particles = new THREE.Points(particleGeo, particleMat);
    scene.add(particles);

    // ── Grid floor ──
    const gridHelper = new THREE.GridHelper(20, 30, 0x1a1d2e, 0x0f1118);
    gridHelper.position.y = -3;
    scene.add(gridHelper);

    // ── Animation ──
    let scrollY = 0;
    window.addEventListener('scroll', () => { scrollY = window.scrollY; });

    function animate() {
        requestAnimationFrame(animate);
        const t = performance.now() * 0.001;

        // Slow rotation
        tower.rotation.y = Math.sin(t * 0.2) * 0.1;
        boom.rotation.y = Math.sin(t * 0.15) * 0.08;
        building.rotation.y = t * 0.05;
        smallBuild.rotation.y = -t * 0.04;

        // Particle drift
        const posArr = particleGeo.attributes.position.array;
        for (let i = 0; i < particleCount; i++) {
            posArr[i * 3 + 1] += Math.sin(t + i) * 0.002;
        }
        particleGeo.attributes.position.needsUpdate = true;

        // Scroll-driven camera movement
        const scrollFactor = scrollY / window.innerHeight;
        camera.position.y = 2 - scrollFactor * 3;
        camera.position.z = 8 + scrollFactor * 4;
        camera.lookAt(0, 0, 0);

        renderer.render(scene, camera);
    }
    animate();

    // ── Resize Handler ──
    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });
})();
