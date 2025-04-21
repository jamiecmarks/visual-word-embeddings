'use client';

import React, { useState, useMemo, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Html, Sparkles, Line } from '@react-three/drei';
import { UMAP } from 'umap-js';
import * as THREE from 'three';

interface Props {
  vocab: string[];
  embeddings: number[][];
}

function WordPoint({
  word,
  position,
  isHighlighted,
}: {
  word: string;
  position: [number, number, number];
  isHighlighted?: boolean;
}) {
  const [x, y, z] = position;

  // Normalize for color mapping
  const normalize = (v: number) => (v + 50) / 100;
  const r = normalize(x);
  const g = normalize(y);
  const b = normalize(z);

  const emissiveColor = new THREE.Color(r * 0.3, g * 0.3, b * 0.6);
  const [noiseFactor, setNoiseFactor] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setNoiseFactor(Math.sin(Date.now() * 0.002) * 0.5); // Slow oscillation for effect
    }, 50);

    return () => clearInterval(interval);
  }, []);

  return (
    <group position={position}>
      <mesh>
        <sphereGeometry args={[0.5, 64, 64]} />
        <meshStandardMaterial
          color={isHighlighted ? new THREE.Color(1, 0.5, 0) : new THREE.Color(0, 0, 0)}
          emissive={emissiveColor}
          emissiveIntensity={0.7}
          transparent
          opacity={0.3 + noiseFactor * 0.4}
          roughness={0.85}
          metalness={0}
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
  const [currentVocab, setCurrentVocab] = useState(vocab);
  const [currentEmbeddings, setCurrentEmbeddings] = useState(embeddings);
  const [inputValue, setInputValue] = useState('');
  const [isInputVisible, setIsInputVisible] = useState(false);
  const [validationError, setValidationError] = useState('');
  const [highlightedWord, setHighlightedWord] = useState<string | null>(null);

  const coords3d = useMemo(() => {
    const umap = new UMAP({ nComponents: 3, nNeighbors: Math.min(3, currentEmbeddings.length - 1), spread: 5, minDist: 0.8 });
    return umap.fit(currentEmbeddings);
  }, [currentEmbeddings]);

  // Function to calculate Euclidean distance between two points
  const calculateDistance = (a: number[], b: number[]) => {
    return Math.sqrt(a.reduce((acc, val, idx) => acc + Math.pow(val - b[idx], 2), 0));
  };

  // Function to find the nearest neighbors of the new word
  const findNearestNeighbors = (embedding: number[], k: number = 5) => {
    const distances = currentEmbeddings.map((e) => calculateDistance(e, embedding));
    const sortedIndexes = distances
      .map((d, i) => ({ index: i, distance: d }))
      .sort((a, b) => a.distance - b.distance)
      .slice(1, k + 1); // Skip the word itself
    return sortedIndexes.map((item) => item.index);
  };

  // Validate the word
  const validateWord = (word: string) => {
    const valid = /^[a-zA-Z]+$/.test(word); // Only letters
    if (!valid) {
      setValidationError('Invalid word. Please use only letters.');
    } else {
      setValidationError('');
    }
    return valid;
  };

  const addWord = (newWord: string, newEmbedding: number[]) => {
    if (validateWord(newWord)) {
      setCurrentVocab((prevVocab) => [...prevVocab, newWord]);
      setCurrentEmbeddings((prevEmbeddings) => [...prevEmbeddings, newEmbedding]);
      setInputValue(''); // Clear the input field
      setIsInputVisible(false); // Hide input after adding
      setHighlightedWord(newWord); // Highlight the new word

      // Find the closest words and update their states to highlight them too
      const nearestNeighbors = findNearestNeighbors(newEmbedding);
      return nearestNeighbors;
    }
    return [];
  };

  const nearestNeighbors = useMemo(() => {
    if (highlightedWord === null) return [];
    const idx = currentVocab.indexOf(highlightedWord);
    if (idx === -1) return [];
    return findNearestNeighbors(currentEmbeddings[idx]);
  }, [highlightedWord, currentEmbeddings, currentVocab]);

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

          {/* Render the WordPoint components */}
          {coords3d.map(([x, y, z], i) => (
            <WordPoint
              key={i}
              word={currentVocab[i]}
              position={[x, y, z]}
              isHighlighted={currentVocab[i] === highlightedWord || nearestNeighbors.includes(i)}
            />
          ))}

          {/* Render connections between the new word and its neighbors */}
          {nearestNeighbors.map((neighborIdx) => {
            const [x1, y1, z1] = coords3d[currentVocab.indexOf(highlightedWord)];
            const [x2, y2, z2] = coords3d[neighborIdx];
            return (
              <Line
                key={neighborIdx}
                points={[[x1, y1, z1], [x2, y2, z2]]}
                color="white"
                lineWidth={2}
                opacity={0.5}
              />
            );
          })}
        </Canvas>
      </div>

      {/* Add Word button */}
      <button
        onClick={() => setIsInputVisible(true)}
        style={{
          position: 'absolute',
          bottom: '20px',
          left: '20px',
          padding: '10px 20px',
          backgroundColor: '#007BFF',
          color: '#fff',
          border: 'none',
          borderRadius: '5px',
          cursor: 'pointer',
        }}
      >
        Add Word
      </button>

      {/* Input box for adding a new word */}
      {isInputVisible && (
        <div style={{ position: 'absolute', bottom: '80px', left: '20px' }}>
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Enter word"
            style={{
              padding: '10px',
              borderRadius: '5px',
              border: '1px solid #ccc',
              width: '200px',
            }}
          />
          <button
            onClick={() => {
              const newEmbedding = [Math.random(), Math.random(), Math.random()]; // Just an example embedding
              addWord(inputValue, newEmbedding);
            }}
            style={{
              padding: '10px 20px',
              backgroundColor: '#28a745',
              color: '#fff',
              border: 'none',
              borderRadius: '5px',
              cursor: 'pointer',
              marginLeft: '10px',
            }}
          >
            Add
          </button>
          <button
            onClick={() => setIsInputVisible(false)}
            style={{
              padding: '10px 20px',
              backgroundColor: '#dc3545',
              color: '#fff',
              border: 'none',
              borderRadius: '5px',
              cursor: 'pointer',
              marginLeft: '10px',
            }}
          >
            Cancel
          </button>

          {/* Validation error message */}
          {validationError && (
            <div style={{ color: 'red', marginTop: '10px' }}>{validationError}</div>
          )}
        </div>
      )}
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
