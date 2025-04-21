'use client';

import React, { useState } from 'react';
import EmbeddingScene from '../../components/EmbeddingScene';
import { useEmbeddings } from '../../hooks/useEmbeddings';
import vocab from '../../public/vocab.json';

export default function Page() {
  const [words] = useState<string[]>(vocab as string[]);
  const embeddings = useEmbeddings(words);

  if (!embeddings) return <div className="p-4">Loading embeddings...</div>;

  return (
    <main className="h-screen bg-gray-900">
      <EmbeddingScene vocab={words} embeddings={embeddings} />
    </main>
  );
}
