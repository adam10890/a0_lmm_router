#!/usr/bin/env perl
# Minimal GPU stats HTTP sidecar for the a0-lmm-net Docker network.
# Exposes GET /health and GET /gpu-stats (nvidia-smi JSON).
# No auth — reachable only inside the compose network, not published to the host.

use strict;
use warnings;
use IO::Socket::INET;

my $host = "0.0.0.0";
my $port = 55502;
for (my $i = 0; $i < @ARGV; $i++) {
    if ($ARGV[$i] eq "--host" && $i + 1 < @ARGV) { $host = $ARGV[++$i]; next; }
    if ($ARGV[$i] eq "--port" && $i + 1 < @ARGV) { $port = int($ARGV[++$i]); next; }
}

sub json_str {
    my ($s) = @_;
    $s //= "";
    $s =~ s/\\/\\\\/g;
    $s =~ s/"/\\"/g;
    $s =~ s/\r//g;
    $s =~ s/\n/\\n/g;
    return qq{"$s"};
}

sub gpu_payload_json {
    my @cmd = (
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    );
    my $out = `@cmd 2>&1`;
    my $code = $? >> 8;
    if ($code != 0) {
        my $err = $out || "nvidia-smi failed ($code)";
        chomp $err;
        return '{"ok":false,"error":' . json_str($err) . ',"gpus":[]}';
    }
    my @items;
    for my $line (split /\n/, $out) {
        $line =~ s/^\s+|\s+$//g;
        next unless length $line;
        my @p = map { s/^\s+|\s+$//gr } split /,/, $line;
        next unless @p >= 7;
        push @items, sprintf(
            '{"id":%d,"name":%s,"total_vram_mb":%d,"used_vram_mb":%d,"free_vram_mb":%d,"utilization_pct":%d,"temperature_c":%d}',
            int($p[0]), json_str($p[1]), int($p[2]), int($p[3]), int($p[4]), int($p[5]), int($p[6]),
        );
    }
    return sprintf('{"ok":true,"source":"fleet_sidecar","count":%d,"gpus":[%s]}', scalar @items, join(",", @items));
}

sub respond {
    my ($client, $code, $body, $ctype) = @_;
    $ctype ||= "application/json; charset=utf-8";
    my $len = length($body);
    print $client "HTTP/1.1 $code\r\nContent-Type: $ctype\r\nContent-Length: $len\r\nConnection: close\r\n\r\n$body";
}

my $server = IO::Socket::INET->new(
    LocalAddr => $host,
    LocalPort => $port,
    Proto     => "tcp",
    Reuse     => 1,
    Listen    => 32,
) or die "bind failed: $!\n";

print "[gpu_stats_sidecar] listening on $host:$port\n";

while (my $client = $server->accept()) {
    eval {
        local $/ = "\r\n";
        my $req = <$client>;
        my $path = "/";
        if (defined $req && $req =~ m{^GET\s+(/[^\s?]*)\s+HTTP}i) {
            $path = $1;
        }
        while (defined(my $line = <$client>)) {
            last if $line eq "\r\n" || $line eq "\n" || $line eq "";
        }
        if ($path eq "/health") {
            respond($client, "200 OK", '{"ok":true,"service":"gpu_stats_sidecar"}');
        } elsif ($path eq "/gpu-stats") {
            my $body = gpu_payload_json();
            my $status = ($body =~ /"ok":true/) ? "200 OK" : "500 Internal Server Error";
            respond($client, $status, $body);
        } else {
            respond($client, "404 Not Found", '{"ok":false,"error":"not found"}');
        }
    };
    close $client;
}
