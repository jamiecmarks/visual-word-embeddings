'use client';

import React, { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Html, Stars, Sparkles } from '@react-three/drei';
import { UMAP } from 'umap-js';
import * as THREE from 'three';

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
function WordPoint({
  word,
  position
}: {
  word: string;
  position: [number, number, number];
}) {
  const texture = new THREE.TextureLoader().load('/planet_texture.jpg'); // Use your planet texture path here
  const [x, y, z] = position;

  // Normalize coordinates to [0, 1]
  const normalize = (v: number) => (v + 50) / 100;
  const r = normalize(x);
  const g = normalize(y);
  const b = normalize(z);

  const baseColor = new THREE.Color(r * 0.5, g * 0.5, b * 0.5); // muted base
  const emissiveColor = new THREE.Color(r * 0.2, g * 0.2, b * 0.3); // subtle glow

  return (
    <group position={position}>
      <mesh>
        <sphereGeometry args={[0.5, 32, 32]} />
        <meshStandardMaterial
          map={texture} // Apply the planet texture
          color={baseColor} // Base color for the planet
          emissive={emissiveColor} // Soft emissive glow for planet effect
          emissiveIntensity={0.15} // Adjust the intensity of the glow
          roughness={0.85} // Give it some roughness to feel more realistic
          metalness={0.1} // Slight metallic look to simulate planet surface
        />
      </mesh>
      <Html distanceFactor={15} position={[0, 0.7, 0]}>
        <div style={{ backgroundColor: "black", color: 'white', fontSize: '1rem', whiteSpace: 'nowrap' }}>{word}</div>
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

