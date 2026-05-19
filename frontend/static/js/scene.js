/**
 * Three.js 3D Scene — Realistic Construction Crane with Scroll Animation
 * CAT yellow/black industrial theme — solid materials with proper lighting
 */
(function () {
    const canvas = document.getElementById('hero-canvas');
    if (!canvas) return;

    /* ── Renderer ── */
    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x050505, 0.008);

    // Responsive camera setup
    const isMobile = window.innerWidth < 768;
    const fov = isMobile ? 60 : 50;
    const camera = new THREE.PerspectiveCamera(fov, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 1, isMobile ? 18 : 16);

    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;

    /* ── Lighting ── */
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.25);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xfff5e0, 0.9);
    dirLight.position.set(8, 15, 10);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.set(1024, 1024);
    dirLight.shadow.camera.near = 0.5;
    dirLight.shadow.camera.far = 50;
    dirLight.shadow.camera.left = -15;
    dirLight.shadow.camera.right = 15;
    dirLight.shadow.camera.top = 15;
    dirLight.shadow.camera.bottom = -15;
    scene.add(dirLight);

    const rimLight = new THREE.DirectionalLight(0xf2c200, 0.3);
    rimLight.position.set(-6, 8, -5);
    scene.add(rimLight);

    const pointLight = new THREE.PointLight(0xf2c200, 0.5, 25);
    pointLight.position.set(0, 8, 3);
    scene.add(pointLight);

    /* ── Materials ── */
    const catYellow = new THREE.MeshStandardMaterial({
        color: 0xf2c200, roughness: 0.45, metalness: 0.3
    });
    const catYellowDark = new THREE.MeshStandardMaterial({
        color: 0xc9a200, roughness: 0.5, metalness: 0.35
    });
    const darkSteel = new THREE.MeshStandardMaterial({
        color: 0x1a1a1a, roughness: 0.3, metalness: 0.8
    });
    const steel = new THREE.MeshStandardMaterial({
        color: 0x333333, roughness: 0.35, metalness: 0.7
    });
    const cabGlass = new THREE.MeshStandardMaterial({
        color: 0x88ccff, roughness: 0.1, metalness: 0.9, transparent: true, opacity: 0.4
    });
    const cableMat = new THREE.MeshStandardMaterial({
        color: 0x555555, roughness: 0.6, metalness: 0.5
    });
    const concreteMat = new THREE.MeshStandardMaterial({
        color: 0x2a2a2a, roughness: 0.9, metalness: 0.05
    });
    const warningMat = new THREE.MeshStandardMaterial({
        color: 0x111111, roughness: 0.5, metalness: 0.3
    });

    /* ── Helper: create lattice cross-bracing for a section ── */
    function createLatticeBrace(width, height, depth, mat) {
        const group = new THREE.Group();
        const barRadius = 0.03;

        // Vertical corner posts
        const postGeo = new THREE.CylinderGeometry(barRadius * 1.5, barRadius * 1.5, height, 6);
        const positions = [
            [-width / 2, 0, -depth / 2],
            [width / 2, 0, -depth / 2],
            [-width / 2, 0, depth / 2],
            [width / 2, 0, depth / 2]
        ];
        positions.forEach(([x, y, z]) => {
            const post = new THREE.Mesh(postGeo, mat);
            post.position.set(x, y, z);
            post.castShadow = true;
            group.add(post);
        });

        // Horizontal braces at top and bottom
        const hBarGeo = new THREE.CylinderGeometry(barRadius, barRadius, width, 5);
        const dBarGeo = new THREE.CylinderGeometry(barRadius, barRadius, depth, 5);
        [height / 2, -height / 2, 0].forEach(y => {
            // Front and back
            [depth / 2, -depth / 2].forEach(z => {
                const bar = new THREE.Mesh(hBarGeo, mat);
                bar.rotation.z = Math.PI / 2;
                bar.position.set(0, y, z);
                group.add(bar);
            });
            // Left and right
            [width / 2, -width / 2].forEach(x => {
                const bar = new THREE.Mesh(dBarGeo, mat);
                bar.rotation.x = Math.PI / 2;
                bar.position.set(x, y, 0);
                group.add(bar);
            });
        });

        // Diagonal cross braces (X pattern) on each face
        const diagLen = Math.sqrt(width * width + height * height);
        const diagGeo = new THREE.CylinderGeometry(barRadius * 0.8, barRadius * 0.8, diagLen, 4);
        const diagAngle = Math.atan2(height, width);

        // Front and back face diagonals
        [depth / 2, -depth / 2].forEach(z => {
            const d1 = new THREE.Mesh(diagGeo, mat);
            d1.rotation.z = diagAngle;
            d1.position.set(0, 0, z);
            group.add(d1);

            const d2 = new THREE.Mesh(diagGeo, mat);
            d2.rotation.z = -diagAngle;
            d2.position.set(0, 0, z);
            group.add(d2);
        });

        return group;
    }

    /* ── Crane Group ── */
    const craneGroup = new THREE.Group();

    // ── Base Platform ──
    const baseGeo = new THREE.BoxGeometry(2.2, 0.3, 2.2);
    const basePlatform = new THREE.Mesh(baseGeo, darkSteel);
    basePlatform.position.set(0, -3, 0);
    basePlatform.castShadow = true;
    basePlatform.receiveShadow = true;
    craneGroup.add(basePlatform);

    // Base bolts
    const boltGeo = new THREE.CylinderGeometry(0.08, 0.08, 0.15, 8);
    [[-0.8, -0.8], [0.8, -0.8], [-0.8, 0.8], [0.8, 0.8]].forEach(([x, z]) => {
        const bolt = new THREE.Mesh(boltGeo, steel);
        bolt.position.set(x, -2.78, z);
        craneGroup.add(bolt);
    });

    // ── Tower (Lattice Sections) ──
    const towerSections = 6;
    const sectionHeight = 1.6;
    const towerWidth = 0.7;
    for (let i = 0; i < towerSections; i++) {
        const section = createLatticeBrace(towerWidth, sectionHeight, towerWidth, catYellow);
        section.position.set(0, -2.85 + 0.3 + sectionHeight / 2 + i * sectionHeight, 0);
        craneGroup.add(section);
    }

    // Tower top cap
    const towerTopY = -2.85 + 0.3 + towerSections * sectionHeight;
    const capGeo = new THREE.BoxGeometry(1.0, 0.2, 1.0);
    const cap = new THREE.Mesh(capGeo, catYellowDark);
    cap.position.set(0, towerTopY, 0);
    cap.castShadow = true;
    craneGroup.add(cap);

    // ── Slewing Unit (Turntable) ──
    const slewGeo = new THREE.CylinderGeometry(0.5, 0.55, 0.35, 16);
    const slew = new THREE.Mesh(slewGeo, darkSteel);
    slew.position.set(0, towerTopY + 0.25, 0);
    slew.castShadow = true;
    craneGroup.add(slew);

    // ── Operator Cab ──
    const cabGroup = new THREE.Group();
    // Cab body
    const cabBodyGeo = new THREE.BoxGeometry(1.1, 0.9, 0.9);
    const cabBody = new THREE.Mesh(cabBodyGeo, catYellow);
    cabBody.position.set(0.3, 0.45, 0);
    cabBody.castShadow = true;
    cabGroup.add(cabBody);

    // Cab roof
    const roofGeo = new THREE.BoxGeometry(1.2, 0.08, 1.0);
    const roof = new THREE.Mesh(roofGeo, catYellowDark);
    roof.position.set(0.3, 0.95, 0);
    cabGroup.add(roof);

    // Cab windows (glass panels)
    const windowGeo = new THREE.PlaneGeometry(0.9, 0.5);
    const frontWindow = new THREE.Mesh(windowGeo, cabGlass);
    frontWindow.position.set(0.86, 0.55, 0);
    frontWindow.rotation.y = Math.PI / 2;
    cabGroup.add(frontWindow);

    const sideWindow = new THREE.Mesh(new THREE.PlaneGeometry(0.7, 0.5), cabGlass);
    sideWindow.position.set(0.3, 0.55, 0.46);
    cabGroup.add(sideWindow);

    const sideWindow2 = sideWindow.clone();
    sideWindow2.position.z = -0.46;
    cabGroup.add(sideWindow2);

    // Cab floor
    const floorGeo = new THREE.BoxGeometry(1.15, 0.06, 0.95);
    const floor = new THREE.Mesh(floorGeo, darkSteel);
    floor.position.set(0.3, 0.02, 0);
    cabGroup.add(floor);

    cabGroup.position.set(0, towerTopY + 0.25, 0);
    craneGroup.add(cabGroup);

    // ── King Post (A-frame mast above slew) ──
    const kingPostGeo = new THREE.CylinderGeometry(0.06, 0.08, 2.4, 6);
    const kingPost = new THREE.Mesh(kingPostGeo, catYellow);
    kingPost.position.set(0, towerTopY + 1.6, 0);
    kingPost.castShadow = true;
    craneGroup.add(kingPost);

    // King post cap
    const kingCapGeo = new THREE.BoxGeometry(0.3, 0.12, 0.3);
    const kingCap = new THREE.Mesh(kingCapGeo, catYellowDark);
    kingCap.position.set(0, towerTopY + 2.85, 0);
    craneGroup.add(kingCap);

    // ── Jib (Main Boom Arm) ──
    const jibGroup = new THREE.Group();
    const jibLength = 9;
    const jibSections = 8;
    const jibSectionLen = jibLength / jibSections;
    for (let i = 0; i < jibSections; i++) {
        const sectionW = 0.25 - i * 0.01;
        const section = createLatticeBrace(sectionW, 0.3, sectionW, catYellow);
        section.rotation.z = Math.PI / 2;
        section.position.set(jibSectionLen / 2 + i * jibSectionLen, 0, 0);
        jibGroup.add(section);
    }

    // Jib bottom chord (solid bar for rigidity look)
    const jibBarGeo = new THREE.BoxGeometry(jibLength, 0.06, 0.06);
    const jibBar = new THREE.Mesh(jibBarGeo, catYellowDark);
    jibBar.position.set(jibLength / 2, -0.15, 0);
    jibGroup.add(jibBar);

    jibGroup.position.set(0.5, towerTopY + 0.9, 0);
    craneGroup.add(jibGroup);

    // ── Counter-Jib (Back arm — full lattice matching main jib) ──
    const counterJibGroup = new THREE.Group();
    const cjibLength = 4;
    const cjibSections = 4;
    const cjibSectionLen = cjibLength / cjibSections;
    for (let i = 0; i < cjibSections; i++) {
        const sectionW = 0.25;
        const section = createLatticeBrace(sectionW, 0.28, sectionW, catYellow);
        section.rotation.z = Math.PI / 2;
        section.position.set(-(cjibSectionLen / 2 + i * cjibSectionLen), 0, 0);
        counterJibGroup.add(section);
    }

    // Counter-jib bottom chord bar (matches main jib style)
    const cjibBarGeo = new THREE.BoxGeometry(cjibLength, 0.06, 0.06);
    const cjibBar = new THREE.Mesh(cjibBarGeo, catYellowDark);
    cjibBar.position.set(-cjibLength / 2, -0.14, 0);
    counterJibGroup.add(cjibBar);

    // Counter-jib top chord bar
    const cjibTopBarGeo = new THREE.BoxGeometry(cjibLength, 0.06, 0.06);
    const cjibTopBar = new THREE.Mesh(cjibTopBarGeo, catYellowDark);
    cjibTopBar.position.set(-cjibLength / 2, 0.14, 0);
    counterJibGroup.add(cjibTopBar);

    counterJibGroup.position.set(0, towerTopY + 0.9, 0);
    craneGroup.add(counterJibGroup);

    // ── Counterweight ──
    const cwGroup = new THREE.Group();
    for (let i = 0; i < 3; i++) {
        const cwBlockGeo = new THREE.BoxGeometry(0.5, 0.55, 0.5);
        const cwBlock = new THREE.Mesh(cwBlockGeo, concreteMat);
        cwBlock.position.set(-3.2 - i * 0.55, -0.1, 0);
        cwBlock.castShadow = true;
        cwGroup.add(cwBlock);
    }
    cwGroup.position.set(0, towerTopY + 0.5, 0);
    craneGroup.add(cwGroup);

    // ── Stay Cables (from king post to jib tip and counter-jib) ──
    function createCable(from, to, mat) {
        const dir = new THREE.Vector3().subVectors(to, from);
        const len = dir.length();
        const cableGeo = new THREE.CylinderGeometry(0.015, 0.015, len, 4);
        const cable = new THREE.Mesh(cableGeo, mat);

        const mid = new THREE.Vector3().addVectors(from, to).multiplyScalar(0.5);
        cable.position.copy(mid);
        cable.lookAt(to);
        cable.rotateX(Math.PI / 2);
        return cable;
    }

    const kingTop = new THREE.Vector3(0, towerTopY + 2.85, 0);
    const jibTip = new THREE.Vector3(jibLength + 0.5, towerTopY + 0.9, 0);
    const cjibEnd = new THREE.Vector3(-cjibLength - 0.3, towerTopY + 0.6, 0);

    craneGroup.add(createCable(kingTop, jibTip, cableMat));
    craneGroup.add(createCable(kingTop, cjibEnd, cableMat));

    // Mid-span stay cable
    const jibMid = new THREE.Vector3(jibLength * 0.5 + 0.5, towerTopY + 0.9, 0);
    craneGroup.add(createCable(kingTop, jibMid, cableMat));

    // ── Trolley (moves along jib) ──
    const trolleyGroup = new THREE.Group();
    const trolleyBodyGeo = new THREE.BoxGeometry(0.35, 0.15, 0.3);
    const trolleyBody = new THREE.Mesh(trolleyBodyGeo, darkSteel);
    trolleyGroup.add(trolleyBody);

    // Trolley wheels
    const wheelGeo = new THREE.CylinderGeometry(0.05, 0.05, 0.05, 8);
    [[-0.12, 0.12], [0.12, 0.12], [-0.12, -0.12], [0.12, -0.12]].forEach(([x, z]) => {
        const wheel = new THREE.Mesh(wheelGeo, steel);
        wheel.rotation.x = Math.PI / 2;
        wheel.position.set(x, 0.1, z);
        trolleyGroup.add(wheel);
    });

    trolleyGroup.position.set(6, towerTopY + 0.9 - 0.25, 0);
    craneGroup.add(trolleyGroup);

    // ── Hoist Cable + Hook ──
    const hoistGroup = new THREE.Group();

    const hoistCableGeo = new THREE.CylinderGeometry(0.012, 0.012, 4, 6);
    const hoistCable = new THREE.Mesh(hoistCableGeo, cableMat);
    hoistCable.position.set(0, -2, 0);
    hoistGroup.add(hoistCable);

    // Hook block
    const hookBlockGeo = new THREE.BoxGeometry(0.2, 0.18, 0.2);
    const hookBlock = new THREE.Mesh(hookBlockGeo, darkSteel);
    hookBlock.position.set(0, -4, 0);
    hookBlock.castShadow = true;
    hoistGroup.add(hookBlock);

    // Hook curve (simplified)
    const hookCurveGeo = new THREE.TorusGeometry(0.08, 0.02, 6, 12, Math.PI * 1.3);
    const hookCurve = new THREE.Mesh(hookCurveGeo, steel);
    hookCurve.position.set(0, -4.2, 0);
    hookCurve.rotation.x = Math.PI * 0.1;
    hoistGroup.add(hookCurve);

    hoistGroup.position.set(6, towerTopY + 0.9, 0);
    craneGroup.add(hoistGroup);

    // ── Warning Stripes on cab ──
    const stripeGeo = new THREE.BoxGeometry(0.02, 0.9, 0.92);
    for (let i = 0; i < 3; i++) {
        const stripe = new THREE.Mesh(stripeGeo, warningMat);
        stripe.position.set(0.3 + (i - 1) * 0.3, towerTopY + 0.7, 0);
        craneGroup.add(stripe);
    }

    // ── Warning beacon light on king post ──
    const beaconGeo = new THREE.SphereGeometry(0.08, 8, 8);
    const beaconMat = new THREE.MeshStandardMaterial({
        color: 0xff3300, emissive: 0xff3300, emissiveIntensity: 0.8,
        roughness: 0.3, metalness: 0.1
    });
    const beacon = new THREE.Mesh(beaconGeo, beaconMat);
    beacon.position.set(0, towerTopY + 3.0, 0);
    craneGroup.add(beacon);

    // Position the whole crane — scaled & lowered for text readability
    const craneScale = isMobile ? 0.55 : 0.85;
    craneGroup.scale.set(craneScale, craneScale, craneScale);
    craneGroup.position.set(isMobile ? -1 : -2, -3, 0);
    scene.add(craneGroup);

    /* ── Background Buildings (solid, distinct colors, visible) ── */
    const buildingColors = [
        { body: 0x1a2a3a, window: 0x66bbff, emissive: 0x3388cc },  // Steel blue
        { body: 0x2a1a2a, window: 0xff88cc, emissive: 0xcc4488 },  // Purple tint
        { body: 0x1a2a1a, window: 0x88ff88, emissive: 0x44aa44 },  // Green tint
        { body: 0x2a2218, window: 0xffcc66, emissive: 0xcc8833 },  // Warm amber
        { body: 0x1a2228, window: 0x66ddcc, emissive: 0x33aa88 },  // Teal
        { body: 0x2a1820, window: 0xff8888, emissive: 0xcc4444 },  // Warm red
    ];

    function createBuilding(w, h, d, x, y, z, colorIdx) {
        const clr = buildingColors[colorIdx % buildingColors.length];
        const bMat = new THREE.MeshStandardMaterial({ color: clr.body, roughness: 0.7, metalness: 0.3 });
        const geo = new THREE.BoxGeometry(w, h, d);
        const mesh = new THREE.Mesh(geo, bMat);
        mesh.position.set(x, y, z);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        scene.add(mesh);

        // Lit windows
        const windowGeo = new THREE.PlaneGeometry(0.15, 0.15);
        const windowMat = new THREE.MeshStandardMaterial({
            color: clr.window, emissive: clr.emissive, emissiveIntensity: 0.7,
            roughness: 0.3, transparent: true, opacity: 0.6
        });
        const floors = Math.floor(h / 0.55);
        const cols = Math.floor(w / 0.45);
        for (let f = 0; f < floors; f++) {
            for (let c = 0; c < cols; c++) {
                if (Math.random() > 0.3) {
                    const win = new THREE.Mesh(windowGeo, windowMat);
                    win.position.set(
                        x - w / 2 + 0.3 + c * 0.45,
                        y - h / 2 + 0.4 + f * 0.55,
                        z + d / 2 + 0.01
                    );
                    scene.add(win);
                }
            }
        }

        // Edge glow trim at top
        const trimGeo = new THREE.BoxGeometry(w + 0.05, 0.04, d + 0.05);
        const trimMat = new THREE.MeshStandardMaterial({
            color: clr.window, emissive: clr.emissive, emissiveIntensity: 0.4,
            roughness: 0.3, transparent: true, opacity: 0.5
        });
        const trim = new THREE.Mesh(trimGeo, trimMat);
        trim.position.set(x, y + h / 2, z);
        scene.add(trim);

        return mesh;
    }

    // Buildings closer and more visible, with varied colors
    const b1 = createBuilding(3, 5, 3, 6, -0.5, -3, 0);
    const b2 = createBuilding(2.2, 3.5, 2, -6, -1.2, -2, 1);
    const b3 = createBuilding(1.8, 7, 1.8, 4, 0.5, -5, 2);
    const b4 = createBuilding(2.5, 4, 2, -4, -1, -4, 3);
    const b5 = createBuilding(2, 5.5, 2, 8, -0.2, -5, 4);
    const b6 = createBuilding(1.5, 3, 1.5, -8, -1.5, -3, 5);

    /* ── Floating Particles (construction sparks/dust) ── */
    const particleCount = 300;
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    const sizes = new Float32Array(particleCount);
    for (let i = 0; i < particleCount; i++) {
        positions[i * 3] = (Math.random() - 0.5) * 28;
        positions[i * 3 + 1] = (Math.random() - 0.5) * 18;
        positions[i * 3 + 2] = (Math.random() - 0.5) * 18;
        const brightness = 0.5 + Math.random() * 0.5;
        colors[i * 3] = 0.95 * brightness;
        colors[i * 3 + 1] = 0.76 * brightness;
        colors[i * 3 + 2] = 0.05 * brightness;
        sizes[i] = 0.02 + Math.random() * 0.04;
    }
    const particleGeo = new THREE.BufferGeometry();
    particleGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    particleGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    const particleMat = new THREE.PointsMaterial({
        size: 0.05, transparent: true, opacity: 0.6,
        vertexColors: true,
        blending: THREE.AdditiveBlending, depthWrite: false,
    });
    const particles = new THREE.Points(particleGeo, particleMat);
    scene.add(particles);

    /* ── Grid Floor ── */
    const gridHelper = new THREE.GridHelper(30, 40, 0x1a1500, 0x0a0a00);
    gridHelper.position.y = -3;
    gridHelper.material.transparent = true;
    gridHelper.material.opacity = 0.35;
    scene.add(gridHelper);

    // Ground plane (receives shadow)
    const groundGeo = new THREE.PlaneGeometry(40, 40);
    const groundMat = new THREE.MeshStandardMaterial({
        color: 0x080808, roughness: 1, metalness: 0
    });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -3.01;
    ground.receiveShadow = true;
    scene.add(ground);

    /* ── Animation Loop ── */
    let scrollY = 0;
    let targetScrollY = 0;
    window.addEventListener('scroll', () => { targetScrollY = window.scrollY; });

    function animate() {
        requestAnimationFrame(animate);
        const t = performance.now() * 0.001;

        // Smooth scroll interpolation
        scrollY += (targetScrollY - scrollY) * 0.08;
        const scrollFactor = scrollY / window.innerHeight;

        // Crane rotation on scroll (smooth)
        craneGroup.rotation.y = scrollFactor * 0.6 + Math.sin(t * 0.12) * 0.03;

        // Trolley moves along jib based on scroll
        const trolleyX = 3 + Math.sin(t * 0.2 + scrollFactor) * 2.5;
        trolleyGroup.position.x = trolleyX;
        hoistGroup.position.x = trolleyX;

        // Hook drops as you scroll + subtle sway
        const hookDrop = Math.min(scrollFactor * 1.5, 2.5);
        const hookSway = Math.sin(t * 0.8) * 0.08;
        hoistGroup.children.forEach(child => {
            // Shift all hoist children down
        });
        hoistCable.scale.y = 0.5 + hookDrop * 0.3;
        hoistCable.position.y = -1 - hookDrop * 0.6;
        hookBlock.position.y = -2 - hookDrop * 1.2;
        hookBlock.position.x = hookSway;
        hookCurve.position.y = -2.2 - hookDrop * 1.2;
        hookCurve.position.x = hookSway;

        // Beacon blink
        beaconMat.emissiveIntensity = 0.3 + Math.abs(Math.sin(t * 3)) * 0.7;

        // Building subtle rotation
        b1.rotation.y = t * 0.005;
        b2.rotation.y = -t * 0.004;
        b3.rotation.y = t * 0.003;
        b4.rotation.y = -t * 0.003;
        b5.rotation.y = t * 0.004;
        b6.rotation.y = -t * 0.005;

        // Particle drift
        const posArr = particleGeo.attributes.position.array;
        for (let i = 0; i < particleCount; i++) {
            posArr[i * 3 + 1] += Math.sin(t * 0.5 + i * 0.3) * 0.002;
            posArr[i * 3] += Math.cos(t * 0.3 + i * 0.7) * 0.001;
        }
        particleGeo.attributes.position.needsUpdate = true;

        // Scroll-driven camera orbit (parallax depth) — responsive
        const camAngle = scrollFactor * 0.25;
        const baseZ = isMobile ? 18 : 16;
        camera.position.x = Math.sin(camAngle) * (isMobile ? 2 : 3);
        camera.position.y = 1 - scrollFactor * 1.0;
        camera.position.z = baseZ + scrollFactor * 2;
        camera.lookAt(0, 0, 0);

        // Point light follows camera a bit
        pointLight.position.x = camera.position.x * 0.5;
        pointLight.position.z = camera.position.z * 0.3;

        renderer.render(scene, camera);
    }
    animate();

    /* ── Resize (responsive) ── */
    window.addEventListener('resize', () => {
        const nowMobile = window.innerWidth < 768;
        camera.fov = nowMobile ? 60 : 50;
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
        const s = nowMobile ? 0.55 : 0.85;
        craneGroup.scale.set(s, s, s);
        craneGroup.position.x = nowMobile ? -1 : -2;
        craneGroup.position.y = -3;
    });
})();
