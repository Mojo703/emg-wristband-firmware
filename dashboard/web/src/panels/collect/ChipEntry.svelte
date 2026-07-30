<script lang="ts">
  // The one optional-text pattern for the setup form: an input wearing chip
  // clothes. Empty and unfocused it shows its "+ label" like a button; focusing
  // it makes it a text entry; blurred with content it keeps the pressed tint.
  // One element throughout — no button/input swap, so focus never gets lost.
  interface Props {
    label: string;
    value: string;
    disabled: boolean;
    ariaLabel: string;
  }

  let { label, value = $bindable(), disabled, ariaLabel }: Props = $props();
</script>

<input
  type="text"
  class="chip"
  data-filled={value.trim() !== '' ? 'true' : 'false'}
  placeholder={label}
  aria-label={ariaLabel}
  bind:value
  {disabled}
/>
