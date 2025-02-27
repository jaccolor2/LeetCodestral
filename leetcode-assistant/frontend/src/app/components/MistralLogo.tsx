import Image from 'next/image';

interface MistralLogoProps {
  size?: number;
  className?: string;
}

export function MistralLogo({ size = 32, className = '' }: MistralLogoProps) {
  return (
    <Image
      src="/assets/mistral-logo.png"
      alt="Mistral AI"
      width={size}
      height={size}
      className={className}
    />
  );
} 