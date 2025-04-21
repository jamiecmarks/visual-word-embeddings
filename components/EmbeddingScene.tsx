'use client';

import React, { useState, useMemo, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Html, Sparkles, Line } from '@react-three/drei';
import { UMAP } from 'umap-js';
import * as THREE from 'three';
import * as use from '@tensorflow-models/universal-sentence-encoder';

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
            fontSize: '2.5rem',
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
  const [isProcessing, setIsProcessing] = useState(false);
  const [model, setModel] = useState<use.UniversalSentenceEncoder | null>(null);
  const [isModelLoading, setIsModelLoading] = useState(false);

  // Load model on component mount
  useEffect(() => {
    async function loadUSEModel() {
      if (!model && !isModelLoading) {
        setIsModelLoading(true);
        try {
          console.log('Loading Universal Sentence Encoder model...');
          const loadedModel = await use.load();
          setModel(loadedModel);
          console.log('Model loaded successfully');
        } catch (error) {
          console.error('Error loading model:', error);
        } finally {
          setIsModelLoading(false);
        }
      }
    }
    
    loadUSEModel();
  }, [model, isModelLoading]);

  const coords3d = useMemo(() => {
    const umap = new UMAP({ 
      nComponents: 3, 
      nNeighbors: Math.min(5, currentEmbeddings.length - 1), 
      spread: 5, 
      minDist: 0.8 
    });
    return umap.fit(currentEmbeddings);
  }, [currentEmbeddings]);

  const cosineDistance = (a: number[], b: number[]) => {
    const dot = a.reduce((acc, val, i) => acc + val * b[i], 0);
    const normA = Math.sqrt(a.reduce((acc, val) => acc + val * val, 0));
    const normB = Math.sqrt(b.reduce((acc, val) => acc + val * val, 0));
    return 1 - dot / (normA * normB);
  };
  
  // Function to find the nearest neighbors of the new word
  const findNearestNeighbors = (embedding: number[], k: number = 5) => {
    const distances = currentEmbeddings.map((e) => cosineDistance(e, embedding));
    const sortedIndexes = distances
      .map((d, i) => ({ index: i, distance: d }))
      .sort((a, b) => a.distance - b.distance)
      .slice(1, k + 1); // Skip the word itself
    return sortedIndexes.map((item) => item.index);
  };

  // Validate the word
  const validateWord = (word: string) => {
    if (!word.trim()) {
      setValidationError('Please enter a word.');
      return false;
    }
    
    const valid = /^[a-zA-Z]+$/.test(word); // Only letters
    if (!valid) {
      setValidationError('Invalid word. Please use only letters.');
    } else {
      setValidationError('');
    }
    return valid;
  };

  const addWord = async (newWord: string) => {
    if (!validateWord(newWord)) return;
    if (!model) {
      setValidationError('Model is still loading. Please try again in a moment.');
      return;
    }

    setIsProcessing(true);
    try {
      // Generate embedding using the pre-trained model
      const embeddings = await model.embed([newWord.toLowerCase()]);
      const embeddingArray = await embeddings.array();
      const newEmbedding = Array.from(embeddingArray[0]);
      
      setCurrentVocab((prevVocab) => [...prevVocab, newWord]);
      setCurrentEmbeddings((prevEmbeddings) => [...prevEmbeddings, newEmbedding]);
      setInputValue(''); // Clear the input field
      setIsInputVisible(false); // Hide input after adding
      setHighlightedWord(newWord); // Highlight the new word
    } catch (error) {
      console.error('Error generating embedding:', error);
      setValidationError('Failed to generate embedding for this word.');
    } finally {
      setIsProcessing(false);
    }
  };

  // If the model isn't loaded yet, use word2vec-like fallback
  const fallbackAddWord = (newWord: string) => {
    if (!validateWord(newWord)) return;
    
    setIsProcessing(true);
    try {
      // Generate a simple word2vec-like embedding (with lower dimensionality)
      // This is just a fallback when TensorFlow.js model fails to load
      const chars = newWord.toLowerCase().split('');
      const charValues = chars.map(c => c.charCodeAt(0) - 96); // a=1, b=2, etc.
      
      // Create a simple embedding based on character positions
      // This is a very simplistic approach but provides some meaning
      const simpleEmbedding = Array(100).fill(0).map((_, i) => {
        const charIndex = i % charValues.length;
        const charValue = charValues[charIndex];
        const position = i / 100;
        return Math.sin(charValue * position) * Math.cos(charValue);
      });
      
      setCurrentVocab((prevVocab) => [...prevVocab, newWord]);
      setCurrentEmbeddings((prevEmbeddings) => [...prevEmbeddings, simpleEmbedding]);
      setInputValue('');
      setIsInputVisible(false);
      setHighlightedWord(newWord);
    } catch (error) {
      console.error('Error with fallback embedding:', error);
      setValidationError('Failed to generate embedding for this word.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleAddWord = () => {
    if (model) {
      addWord(inputValue);
    } else {
      fallbackAddWord(inputValue);
    }
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
      {isModelLoading ? (
        <div style={{ position: 'absolute', top: '10px', right: '10px', background: 'rgba(0,0,0,0.7)', color: 'white', padding: '5px 10px', borderRadius: '5px' }}>
          Loading embedding model...
        </div>
      ) : model ? (
        <div style={{ position: 'absolute', top: '10px', right: '10px', background: 'rgba(0,100,0,0.7)', color: 'white', padding: '5px 10px', borderRadius: '5px' }}>
          Using Universal Sentence Encoder
        </div>
      ) : (
        <div style={{ position: 'absolute', top: '10px', right: '10px', background: 'rgba(100,0,0,0.7)', color: 'white', padding: '5px 10px', borderRadius: '5px' }}>
          Using fallback embeddings
        </div>
      )}
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
          {highlightedWord && nearestNeighbors.map((neighborIdx) => {
            const highlightedIndex = currentVocab.indexOf(highlightedWord);
            if (highlightedIndex >= 0 && coords3d[highlightedIndex] && coords3d[neighborIdx]) {
              const [x1, y1, z1] = coords3d[highlightedIndex];
              const [x2, y2, z2] = coords3d[neighborIdx];
              return (
                <Line
                  key={`line-${neighborIdx}`}
                  points={[[x1, y1, z1], [x2, y2, z2]]}
                  color="white"
                  lineWidth={2}
                  opacity={0.5}
                />
              );
            }
            return null;
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
            disabled={isProcessing}
          />
          <button
            onClick={handleAddWord}
            style={{
              padding: '10px 20px',
              backgroundColor: '#28a745',
              color: '#fff',
              border: 'none',
              borderRadius: '5px',
              cursor: 'pointer',
              marginLeft: '10px',
            }}
            disabled={isProcessing}
          >
            {isProcessing ? 'Processing...' : 'Add'}
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
            disabled={isProcessing}
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