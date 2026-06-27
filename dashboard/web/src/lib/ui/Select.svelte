<script lang="ts">
  // Thin adapter over the shadcn-svelte Select so panels keep a simple, controlled
  // API (value + onChange), which suits server-derived values that can't be bound.
  import * as Select from '$lib/components/ui/select/index.js';

  interface Option {
    readonly value: string;
    readonly label: string;
  }
  interface Props {
    value: string;
    options: readonly Option[];
    onChange?: (value: string) => void;
    placeholder?: string;
    title?: string;
  }
  let { value, options, onChange, placeholder = 'Select…', title }: Props = $props();

  const selectedLabel = $derived(options.find((option) => option.value === value)?.label ?? '');
</script>

<Select.Root type="single" {value} onValueChange={(next) => onChange?.(next)}>
  <Select.Trigger {title}>{selectedLabel || placeholder}</Select.Trigger>
  <Select.Content>
    {#each options as option (option.value)}
      <Select.Item value={option.value} label={option.label} />
    {/each}
  </Select.Content>
</Select.Root>
