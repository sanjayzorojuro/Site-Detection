/**
 * Three.js 3D Scene — Construction Crane with Scroll Animation
 * CAT yellow/black theme — crane rotates and camera moves on scroll
 */
(function () {
    const canvas = document.getElementById('hero-canvas');
    if (!canvas) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 3, 12);

    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0);

    // CAT yellow color
    const yellowCol = 0xf2c200;
    const yellowDim = 0x665200;
    const darkBorder = 0x1a1a1a;

    // ── Materials ──
    const craneMat = new THREE.LineBasicMaterial({ color: yellowCol, transparent: true, opacity: 0.35 });
    const craneMatBright = new THREE.LineBasicMaterial({ color: yellowCol, transparent: true, opacity: 0.55 });
    const buildMat = new THREE.LineBasicMaterial({ color: yellowDim, transparent: true, opacity: 0.15 });
    const buildMat2 = new THREE.LineBasicMaterial({ color: yellowCol, transparent: true, opacity: 0.1 });

    // ── Construction Crane ──
    const craneGroup = new THREE.Group();

    // Vertical tower (tall)
    const towerGeo = new THREE.BoxGeometry(0.8, 9, 0.8);
    const towerEdges = new THREE.EdgesGeometry(towerGeo);
    const tower = new THREE.LineSegments(towerEdges, craneMatBright);
    tower.position.set(0, 1.5, 0);
    craneGroup.add(tower);

    // Tower cab (operator cabin)
    const cabGeo = new THREE.BoxGeometry(1.4, 1, 1.2);
    const cabEdges = new THREE.EdgesGeometry(cabGeo);
    const cab = new THREE.LineSegments(cabEdges, craneMatBright);
    cab.position.set(0.3, 5.8, 0);
    craneGroup.add(cab);

    // Horizontal boom (jib)
    const boomGeo = new THREE.BoxGeometry(10, 0.35, 0.35);
    const boomEdges = new THREE.EdgesGeometry(boomGeo);
    const boom = new THREE.LineSegments(boomEdges, craneMatBright);
    boom.position.set(3, 6.2, 0);
    craneGroup.add(boom);

    // Counter-jib (shorter back arm)
    const counterGeo = new THREE.BoxGeometry(3.5, 0.3, 0.3);
    const counterEdges = new THREE.EdgesGeometry(counterGeo);
    const counterBoom = new THREE.LineSegments(counterEdges, craneMat);
    counterBoom.position.set(-3, 6.2, 0);
    craneGroup.add(counterBoom);

    // Counterweight
    const cwGeo = new THREE.BoxGeometry(1, 0.8, 0.6);
    const cwEdges = new THREE.EdgesGeometry(cwGeo);
    const cw = new THREE.LineSegments(cwEdges, craneMatBright);
    cw.position.set(-4.5, 5.8, 0);
    craneGroup.add(cw);

    // Top mast (king post)
    const mastGeo = new THREE.BoxGeometry(0.15, 2, 0.15);
    const mastEdges = new THREE.EdgesGeometry(mastGeo);
    const mast = new THREE.LineSegments(mastEdges, craneMat);
    mast.position.set(0, 7.2, 0);
    craneGroup.add(mast);

    // Cable from boom tip
    const cableGeo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(7, 6.2, 0),
        new THREE.Vector3(7, 2.5, 0),
    ]);
    const cable = new THREE.Line(cableGeo, craneMatBright);
    craneGroup.add(cable);

    // Hook
    const hookGeo = new THREE.BoxGeometry(0.4, 0.4, 0.4);
    const hookEdges = new THREE.EdgesGeometry(hookGeo);
    const hook = new THREE.LineSegments(hookEdges, craneMatBright);
    hook.position.set(7, 2.3, 0);
    craneGroup.add(hook);

    // Support cables (stays)
    const stay1Geo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 8.2, 0),
        new THREE.Vector3(7, 6.2, 0),
    ]);
    craneGroup.add(new THREE.Line(stay1Geo, craneMat));

    const stay2Geo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 8.2, 0),
        new THREE.Vector3(-4.5, 6.2, 0),
    ]);
    craneGroup.add(new THREE.Line(stay2Geo, craneMat));

    craneGroup.position.set(-2, -3, 0);
    scene.add(craneGroup);

    // ── Building Wireframes ──
    const build1Geo = new THREE.BoxGeometry(3, 4.5, 3);
    const build1 = new THREE.LineSegments(new THREE.EdgesGeometry(build1Geo), buildMat);
    build1.position.set(5, -0.75, -3);
    scene.add(build1);

    const build2Geo = new THREE.BoxGeometry(2, 2.5, 2);
    const build2 = new THREE.LineSegments(new THREE.EdgesGeometry(build2Geo), buildMat2);
    build2.position.set(-5, -1.75, -2);
    scene.add(build2);

    const build3Geo = new THREE.BoxGeometry(1.8, 6, 1.8);
    const build3 = new THREE.LineSegments(new THREE.EdgesGeometry(build3Geo), buildMat);
    build3.position.set(3, 0, -5);
    scene.add(build3);

    // ── Floating Particles (construction sparks/dust) ──
    const particleCount = 250;
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount; i++) {
        positions[i * 3] = (Math.random() - 0.5) * 24;
        positions[i * 3 + 1] = (Math.random() - 0.5) * 16;
        positions[i * 3 + 2] = (Math.random() - 0.5) * 16;
        // Yellow/amber particle colors
        const r = 0.8 + Math.random() * 0.2;
        const g = 0.6 + Math.random() * 0.2;
        const b = Math.random() * 0.1;
        colors[i * 3] = r;
        colors[i * 3 + 1] = g;
        colors[i * 3 + 2] = b;
    }
    const particleGeo = new THREE.BufferGeometry();
    particleGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    particleGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    const particleMat = new THREE.PointsMaterial({
        size: 0.045, transparent: true, opacity: 0.55,
        vertexColors: true,
        blending: THREE.AdditiveBlending, depthWrite: false,
    });
    const particles = new THREE.Points(particleGeo, particleMat);
    scene.add(particles);

    // ── Grid Floor ──
    const gridHelper = new THREE.GridHelper(24, 36, 0x1a1a1a, 0x0f0f0f);
    gridHelper.position.y = -3;
    scene.add(gridHelper);

    // ── Animation Loop ──
    let scrollY = 0;
    window.addEventListener('scroll', () => { scrollY = window.scrollY; });

    function animate() {
        requestAnimationFrame(animate);
        const t = performance.now() * 0.001;
        const scrollFactor = scrollY / window.innerHeight;

        // Crane rotation on scroll
        craneGroup.rotation.y = scrollFactor * 0.8 + Math.sin(t * 0.15) * 0.05;

        // Hook drops as you scroll
        const hookDrop = Math.min(scrollFactor * 2, 3);
        hook.position.y = 2.3 - hookDrop;
        const cablePositions = cable.geometry.attributes.position.array;
        cablePositions[4] = 2.5 - hookDrop;
        cable.geometry.attributes.position.needsUpdate = true;

        // Building rotation
        build1.rotation.y = t * 0.03;
        build2.rotation.y = -t * 0.025;
        build3.rotation.y = t * 0.02;

        // Particle drift
        const posArr = particleGeo.attributes.position.array;
        for (let i = 0; i < particleCount; i++) {
            posArr[i * 3 + 1] += Math.sin(t + i * 0.5) * 0.002;
            posArr[i * 3] += Math.cos(t * 0.5 + i) * 0.001;
        }
        particleGeo.attributes.position.needsUpdate = true;

        // Scroll-driven camera orbit
        camera.position.x = Math.sin(scrollFactor * 0.3) * 3;
        camera.position.y = 3 - scrollFactor * 2;
        camera.position.z = 12 + scrollFactor * 3;
        camera.lookAt(0, 1, 0);

        renderer.render(scene, camera);
    }
    animate();

    // ── Resize ──
    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });
})();
