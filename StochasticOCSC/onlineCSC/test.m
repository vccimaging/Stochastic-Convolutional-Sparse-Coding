x = [ 50 70 90 110 130 150 170 190];
z1 = 1.0e+04 *[ 1.5084    1.3440    1.2855    1.2529    1.2334    1.2211 ...
    1.2129    1.2069    1.2024    1.1986];
z2 = 1.0e+04 *[ 1.5237    1.3474    1.2961    1.2709    1.2581    1.2503 ...
    1.2468    1.2430    1.2382    1.2374];
z3 = 1.0e+04 *[ 1.5942    1.3897    1.3285    1.2949    1.2783    1.2650 ...
    1.2596    1.2561    1.2525    1.2504];
z4 = 1.0e+04 *[ 1.6369    1.4431    1.3863    1.3632    1.3476    1.3336 ...
    1.3262    1.3213    1.3175    1.3177];
z5 = 1.0e+04 *[ 1.7171    1.5316    1.4932    1.4697    1.4537    1.4431 ...
    1.4363    1.4327    1.4263    1.4224];
z6 = 1.0e+06 *[ 3.4200    0.9877    0.2242    0.0461    0.0225    0.0186 ...
    0.0201    0.0160    0.0141    0.0133  0.0130    0.0127    0.0127];

figure
plot( 1:10, z1, '--o','MarkerSize', 12, 'LineWidth',2 );
hold on
plot( 1:10, z2, '--d','MarkerSize', 12, 'LineWidth',2);
hold on
plot( 1:10, z3, '--^','MarkerSize', 12, 'LineWidth',2 );
hold on
plot( 1:10, z4, '--*','MarkerSize', 12, 'LineWidth',2 );
hold on
plot( 1:10, z5, '--+','MarkerSize', 12, 'LineWidth',2 );
hold on
plot( 1:10, z6(1:10), '--p','MarkerSize', 12, 'LineWidth',2 );
legend('P=1', 'P=0.5', 'P=0.2', 'P=0.1', 'P=0.05', 'Heide et al' );

set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')

set(gca,'fontsize',32)
set(gca, 'XLim', [10 1300]);
% set(gca,'xtick',[50 80 110]);
set(gca, 'YLim', [1.1e4 3e4]);
set(gca,'ytick',[ 1.2e4 2e4 3e4]);
xlabel('Time (s)', 'fontsize',32 );
ylabel('Objective', 'fontsize',32 );

set(gca, 'XLim', [1 10]);
set(gca,'xtick',[1 5 10]);
xlabel('Iterations ', 'fontsize',32 );

fig = gcf;
fig.PaperPositionMode = 'auto'
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
print(fig,'../../nips2018/figure/iteVSobj','-dpdf')

figure
plot( 130:130:1300, z1, '--o','MarkerSize', 12, 'LineWidth',2 );
hold on
plot( 95:95:950, z2, '--d','MarkerSize', 12, 'LineWidth',2);
hold on
plot( 42:42:420, z3, '--^','MarkerSize', 12, 'LineWidth',2 );
hold on
plot( 24:24:240, z4, '--*','MarkerSize', 12, 'LineWidth',2 );
hold on
plot( 17:17:170, z5, '--+','MarkerSize', 12, 'LineWidth',2 );
hold on
plot( 55:55:55*13, z6, '--p','MarkerSize', 12, 'LineWidth',2 );
legend('P=1', 'P=0.5', 'P=0.2', 'P=0.1', 'P=0.05', 'Heide et al' );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
