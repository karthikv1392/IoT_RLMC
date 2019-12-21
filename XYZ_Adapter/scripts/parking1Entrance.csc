set mod 0
set counter 0
set sendCounter 0
loop
if($mod==0)
	areadsensor var
	rdata $var t x sensorVal1
	plus counter $sensorVal1 $counter
	data p s34 $sensorVal1
	function y adapter p1en,100,10
	if($y==10.0)
		send $p 33
	end
	if($y==20.0)
		send $p 11
	end
	if($y==30.0)
		send $p 47
	end
	while($sensorVal1<10.0)
		areadsensor var
		rdata $var t x sensorVal1
		plus counter $sensorVal1 $counter
		data p s34 $sensorVal1
		function y adapter p1en,100,10
		if($y==10.0)
			plus sendCounter $sendCounter 1
			if($counter>=200.0)
				send N 35
			else
				send A 35
			end
			send $p 33
			if ($sendCounter >= 1)
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
		delay 60000
	end
	if($sensorVal1>=10.0)
		set mod 1
	end
end
if($mod==1)
	while($sensorVal1>=10.0)
		areadsensor var
		rdata $var t x sensorVal1
		data p s34 $sensorVal1
		plus counter $sensorVal1 $counter
		function y adapter p1en,100,10
		if($y==10.0)
			plus sendCounter $sendCounter 1
			if($counter>=200.0)
				send N 35
			else
				send A 35
			end
			send $p 33
			if ($sendCounter>= 6)
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
	if($sensorVal1<10.0)
		set mod 0
	end
end
