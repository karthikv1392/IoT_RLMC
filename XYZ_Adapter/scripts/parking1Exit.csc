set mod 0
set counter 0
set sendCounter 0
loop
if($mod==0)
	areadsensor var
	rdata $var t x sensorVal1
	plus counter $sensorVal1 $counter
	data p s33 $sensorVal1
	function y adapter p1ex,100,10
	if($y==10.0)
		send $p 34
	end
	if($y==20.0)
		send $p 11
	end
	if($y==30.0)
		send $p 47
	end
	while($sensorVal1<5.0)
		areadsensor var
		rdata $var t x sensorVal1
		plus counter $sensorVal1 $counter
		data p s33 $sensorVal1
		function y adapter p1ex,100,10
		if($y==10.0)
			plus sendCounter $sendCounter 1
			if($counter>=0)
				send A 35
			else
				send N 35
			end
			send $p 34
			if($sendCounter >= 2)
				send $p 11
				set sendCounter 0
			end
		end
		if($y==20.0)
			send $p 11
		end
		if ($y==30.0)
			send $p 47
		end
		delay 30000
	end
	if($sensorVal1>=5.0)
		set mod 1
	end
end
if($mod==1)
	while($sensorVal1>=5.0)
		areadsensor var
		rdata $var t x sensorVal1
		plus counter $sensorVal1 $counter
		data p s33 $sensorVal1
		function y adapter p1ex,100,10
		if($y==10.0)
			plus sendCounter $sendCounter 1
			if($counter>=0)
				send A 35
			else
				send N 35
			end
		send $p 34
		if($sendCounter >= 6)
			send $p 11
			set sendCounter 0
		end
		end
		if($y==20.0)
			send $p 11
		end
		if ($y==30.0)
			send $p 47
		end
		delay 10000
	end
	if($sensorVal1<5.0)
		set mod 0
	end
end
