import { mount } from 'svelte';
import './app.css';
import App from './App.svelte';

const target = document.getElementById('app');
if (target === null) {
  throw new Error('Application mount point #app not found');
}

export default mount(App, { target });
