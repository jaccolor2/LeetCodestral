import Image from 'next/image';

interface MistralLogoProps {
  size?: number;
  className?: string;
  variant?: 'navbar' | 'chat';
}

export function MistralLogo({ size = 18, className = '', variant = 'chat' }: MistralLogoProps) {
  const imageSrc = variant === 'navbar' ? '/assets/mistral-logo.png' : '/assets/chat-icon.svg';
  
  return (
    <Image
      src={imageSrc}
      alt="Mistral AI"
      width={size}
      height={size}
      className={`logo-zoom ${className}`}
    />
  );
} 