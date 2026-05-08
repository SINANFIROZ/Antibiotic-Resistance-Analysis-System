import { render, screen } from '@testing-library/react';

import { StatCard } from '@/components/stat-card';

describe('StatCard', () => {
  it('renders core metric content', () => {
    render(<StatCard label="Total predictions" value="128" delta="+12%" />);

    expect(screen.getByText('Total predictions')).toBeInTheDocument();
    expect(screen.getByText('128')).toBeInTheDocument();
    expect(screen.getByText('+12%')).toBeInTheDocument();
  });
});
