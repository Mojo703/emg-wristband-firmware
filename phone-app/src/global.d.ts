declare module '*.svelte' {
  import type { Component } from 'svelte';
  const component: Component;
  export default component;
}

declare module '*.css' {
  const content: string;
  export default content;
}
