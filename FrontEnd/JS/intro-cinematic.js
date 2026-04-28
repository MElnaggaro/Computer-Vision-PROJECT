// ============================================
// CINEMATIC INTRO SYSTEM
// ============================================
// Manages the fullscreen intro overlay:
//   1. Ambient particle animation
//   2. Loading state → Ready state
//   3. Click-to-start interaction
//   4. Model reveal + title sequence
//   5. Smooth exit transition
// ============================================

const IntroSystem = (() => {
    // ─── DOM References ───
    const introScreen   = document.getElementById('intro-screen');
    const introCanvas   = document.getElementById('intro-particles');
    const introStatus   = document.getElementById('intro-status');
    const introTitle    = document.getElementById('intro-title');
    const introLoader   = document.getElementById('intro-loader');
    const introCtx      = introCanvas.getContext('2d');

    // ─── State ───
    let modelLoaded  = false;
    let introExited  = false;
    let introParticles = [];
    const PARTICLE_COUNT = 80;

    // Lock scroll during intro
    document.body.classList.add('intro-active');

    // ─── Intro Particle System ───
    class IntroParticle {
        constructor() {
            this.reset();
        }

        reset() {
            this.x = Math.random() * introCanvas.width;
            this.y = Math.random() * introCanvas.height;
            this.size = Math.random() * 2.5 + 0.5;
            this.speedX = (Math.random() - 0.5) * 0.4;
            this.speedY = (Math.random() - 0.5) * 0.4;
            this.opacity = Math.random() * 0.5 + 0.1;
            // Random blue/purple color
            this.hue = Math.random() > 0.5 ? 240 : 270;
            this.saturation = 60 + Math.random() * 30;
            this.lightness  = 50 + Math.random() * 20;
        }

        update() {
            this.x += this.speedX;
            this.y += this.speedY;

            // Wrap around edges
            if (this.x < 0) this.x = introCanvas.width;
            if (this.x > introCanvas.width) this.x = 0;
            if (this.y < 0) this.y = introCanvas.height;
            if (this.y > introCanvas.height) this.y = 0;
        }

        draw() {
            introCtx.beginPath();
            introCtx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
            introCtx.fillStyle = `hsla(${this.hue}, ${this.saturation}%, ${this.lightness}%, ${this.opacity})`;
            introCtx.fill();

            // Glow effect
            introCtx.beginPath();
            introCtx.arc(this.x, this.y, this.size * 3, 0, Math.PI * 2);
            introCtx.fillStyle = `hsla(${this.hue}, ${this.saturation}%, ${this.lightness}%, ${this.opacity * 0.15})`;
            introCtx.fill();
        }
    }

    // ─── Resize Canvas ───
    const resizeIntroCanvas = () => {
        introCanvas.width = window.innerWidth;
        introCanvas.height = window.innerHeight;
    };
    window.addEventListener('resize', resizeIntroCanvas);
    resizeIntroCanvas();

    // ─── Init Particles ───
    const initIntroParticles = () => {
        introParticles = [];
        for (let i = 0; i < PARTICLE_COUNT; i++) {
            introParticles.push(new IntroParticle());
        }
    };

    // ─── Draw Connections ───
    const drawConnections = () => {
        for (let i = 0; i < introParticles.length; i++) {
            for (let j = i + 1; j < introParticles.length; j++) {
                const dx = introParticles[i].x - introParticles[j].x;
                const dy = introParticles[i].y - introParticles[j].y;
                const distance = Math.sqrt(dx * dx + dy * dy);

                if (distance < 120) {
                    introCtx.beginPath();
                    introCtx.moveTo(introParticles[i].x, introParticles[i].y);
                    introCtx.lineTo(introParticles[j].x, introParticles[j].y);
                    introCtx.strokeStyle = `rgba(79, 70, 229, ${0.06 * (1 - distance / 120)})`;
                    introCtx.lineWidth = 0.5;
                    introCtx.stroke();
                }
            }
        }
    };

    // ─── Animate Intro Particles ───
    let introAnimId;
    const animateIntroParticles = () => {
        if (introExited) return;

        introCtx.clearRect(0, 0, introCanvas.width, introCanvas.height);

        // Central radial gradient for depth
        const grad = introCtx.createRadialGradient(
            introCanvas.width / 2, introCanvas.height / 2, 0,
            introCanvas.width / 2, introCanvas.height / 2, introCanvas.width * 0.6
        );
        grad.addColorStop(0, 'rgba(79, 70, 229, 0.04)');
        grad.addColorStop(0.5, 'rgba(147, 51, 234, 0.02)');
        grad.addColorStop(1, 'rgba(5, 5, 5, 0)');
        introCtx.fillStyle = grad;
        introCtx.fillRect(0, 0, introCanvas.width, introCanvas.height);

        introParticles.forEach(p => {
            p.update();
            p.draw();
        });

        drawConnections();

        introAnimId = requestAnimationFrame(animateIntroParticles);
    };

    // ─── Start ───
    initIntroParticles();
    animateIntroParticles();


    // ─── Public: Signal that the 3D model has loaded ───
    const onModelLoaded = () => {
        modelLoaded = true;

        // Transition from loading → ready
        gsap.to('.intro-spinner', {
            opacity: 0,
            scale: 0.5,
            duration: 0.6,
            onComplete: () => {
                document.querySelector('.intro-spinner').style.display = 'none';
            }
        });

        // Update status text
        setTimeout(() => {
            introStatus.textContent = '[ SYSTEM READY — CLICK TO START ]';
            introStatus.classList.add('ready');

            // Enable click
            introScreen.style.cursor = 'pointer';
            introScreen.addEventListener('click', startExperience, { once: true });
        }, 400);
    };


    // ─── Click-to-Start Sequence ───
    const startExperience = () => {
        if (introExited) return;
        introExited = true;

        // 1) Create and play a subtle AI ambient pulse (Web Audio API)
        try {
            const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            const osc = audioCtx.createOscillator();
            const gainNode = audioCtx.createGain();
            osc.type = 'sine';
            osc.frequency.setValueAtTime(180, audioCtx.currentTime);
            osc.frequency.exponentialRampToValueAtTime(60, audioCtx.currentTime + 2);
            gainNode.gain.setValueAtTime(0.08, audioCtx.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 2.5);
            osc.connect(gainNode);
            gainNode.connect(audioCtx.destination);
            osc.start();
            osc.stop(audioCtx.currentTime + 2.5);
        } catch(e) {
            // Audio not supported, continue silently
        }

        // 2) Build the reveal timeline
        const tl = gsap.timeline();

        // Hide status
        tl.to(introStatus, {
            opacity: 0,
            duration: 0.4,
            ease: 'power2.in'
        });

        // 3) Reveal the title
        tl.to(introTitle, {
            opacity: 1,
            y: 0,
            duration: 1.2,
            ease: 'power3.out'
        }, '+=0.3');

        // 4) Signal 3D model to fade in (dispatches custom event)
        tl.call(() => {
            window.dispatchEvent(new CustomEvent('intro:reveal-model'));
        }, null, '-=0.8');

        // 5) Hold
        tl.to({}, { duration: 1.5 });

        // 6) Exit overlay — fade out
        tl.to(introScreen, {
            opacity: 0,
            duration: 1.5,
            ease: 'power2.inOut',
            onComplete: () => {
                introScreen.style.display = 'none';
                document.body.classList.remove('intro-active');
                cancelAnimationFrame(introAnimId);

                // Dispatch event so rest of the site can initialize animations
                window.dispatchEvent(new CustomEvent('intro:complete'));
            }
        });
    };


    // ─── Public API ───
    return {
        onModelLoaded
    };

})();

// Expose globally so three-scene.js can call it
window.IntroSystem = IntroSystem;
