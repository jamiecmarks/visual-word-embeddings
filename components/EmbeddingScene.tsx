'use client';

import React, { useState, useRef, useMemo, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Html, Stars, Sparkles } from '@react-three/drei';
import { UMAP } from 'umap-js';
import * as THREE from 'three';
import { EffectComposer, Bloom } from '@react-three/postprocessing';

interface Props {
  vocab: string[];
  embeddings: number[][];
}

// export default function EmbeddingScene1({ vocab, embeddings }: Props) {
//   // Run UMAP once on the client
//   const coords3d = useMemo(() => {
//     const umap = new UMAP({ nComponents: 3, nNeighbors: Math.min(3, embeddings.length - 1) });
//     return umap.fit(embeddings);
//   }, [embeddings]);
//
//   if (!coords3d) return <div>Computing layout...</div>;
//
//   return (
//     <Canvas camera={{ position: [0, 0, 5], fov: 60 }}>
//       <ambientLight intensity={1} />
//       <directionalLight position={[5, 5, 5]} />
//       {coords3d.map(([x, y, z], i) => (
//         <mesh key={i} position={[x, y, z]}>
//           <sphereGeometry args={[0.03, 16, 16]} />
//           <meshStandardMaterial color="white" />
//           <Html distanceFactor={10}>
//             <div className="text-xs text-white">{vocab[i]}</div>
//           </Html>
//         </mesh>
//       ))}
//       <OrbitControls
//         makeDefault
//         enableZoom={true}
//         zoomSpeed={0.5}
//         minDistance={1}
//         maxDistance={20}
//       />
//     </Canvas>
//   );
// }
//
//
function WordPoint({
  word,
  position,
}: {
  word: string;
  position: [number, number, number];
}) {
  const [x, y, z] = position;
  const meshRef = useRef<THREE.Mesh>(null);

  // Normalize for color mapping
  const normalize = (v: number) => (v + 50) / 100;
  const r = normalize(x);
  const g = normalize(y);
  const b = normalize(z);

  // Dynamic emissive color
  const emissiveColor = new THREE.Color(r * 0.3, g * 0.3, b * 0.6);

  // Dissolving effect: time-based noise
  const [noiseFactor, setNoiseFactor] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setNoiseFactor(Math.sin(Date.now() * 0.002) * 0.5); // Slow oscillation for effect
    }, 50);

    return () => clearInterval(interval); // Cleanup on unmount
  }, []);

  return (
    <group position={position}>
      <mesh ref={meshRef}>
        {/* Dissolving mass using a sphere geometry with subtle noise */}
        <sphereGeometry args={[0.5, 64, 64]} />
        <meshStandardMaterial
          color={new THREE.Color(0, 0, 0)} // Black base color
          emissive={emissiveColor} // Subtle glowing effect
          emissiveIntensity={0.7} // Intensity of the glow
          transparent
          opacity={0.3 + noiseFactor * 0.4} // Low opacity, fluctuating over time
          roughness={0.85} // Diffuse, no gloss
          metalness={0} // Not metallic
          // Create a dissolving effect with animated noise
          displacementMap={new THREE.TextureLoader().load('/noise_texture.jpg')} // Add a noise texture to create variation
          displacementScale={0.1} // Small scale for smooth displacement
        />
      </mesh>

      <Html distanceFactor={20} position={[0, 0.6, 0]}>
        <div
          style={{
            backgroundColor: 'rgba(0,0,0,0.5)',
            padding: '0.1rem 0.4rem',
            borderRadius: '0.25rem',
            color: 'white',
            fontSize: '0.85rem',
            whiteSpace: 'nowrap',
            backdropFilter: 'blur(4px)',
          }}
        >
          {word}
        </div>
      </Html>
    </group>
  );
}

export default function EmbeddingScene({ vocab, embeddings }: Props) {
  const coords3d = useMemo(() => {
    const umap = new UMAP({ nComponents: 3, nNeighbors: Math.min(3, embeddings.length - 1), spread: 5, minDist: 0.8 });
    return umap.fit(embeddings);
  }, [embeddings]);

  if (!coords3d) return <div>Computing layout...</div>;

  return (
    <div>
      <h2 className="text-5xl sm:text-4xl text-center leading-tight">
        3D Word Embeddings
      </h2>
      <div className="center h-full w-full">
        <Canvas
          className="h-full w-full"
          camera={{ position: [0, 0, 100], fov: 75 }}
          style={{ background: "#0d1117" }}
        >
          <color attach="background" args={["#0d1117"]} />
          <fog attach="fog" args={["#0d1117", 60, 200]} />
          <ambientLight intensity={0.6} />
          <directionalLight position={[10, 10, 10]} intensity={0.8} />
          <SpaceDust />
          <OrbitControls enableZoom={true} zoomSpeed={0.5} minDistance={5} maxDistance={200} />
          {coords3d.map(([x, y, z], i) => (
            <WordPoint
              key={i}
              word={vocab[i]}
              position={[x, y, z]}
            />
          ))}
        </Canvas>
      </div>
    </div>
  );
}

function SpaceDust() {
  return (
    <Sparkles
      count={500}
      size={5}
      speed={1}
      scale={[100, 100, 100]}
      color="white"
      position={[0, 0, 0]}
    />
  );
}

