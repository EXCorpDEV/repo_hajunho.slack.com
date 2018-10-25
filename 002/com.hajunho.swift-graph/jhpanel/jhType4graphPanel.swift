//
//  jhType4graphPanel.swift
//  com.hajunho.swift-graph
//
//  Created by Junho HA on 2018. 10. 25..
//  Copyright © 2018년 hajunho.com. All rights reserved.
//

import UIKit

class jhType4graph<T> : jhPanel<T> {
    
    var panelID: Int = 0
    
    var pointCloud = Array<CGPoint>()
    
    override func drawDatas() {
        //        worldEllipse(context: mContext, 100, 100, 100, 100, 2, UIColor.blue.cgColor)
        
        var fx, fy : CGFloat
        
        var x : Int = 0
        for y in (jhDataCenter.mDatas[panelID]?.d)! {
            //ref:drawLine(CGFloat(x)*axisDistance + mMargin, mMargin, CGFloat(x) * axisDistance + mMargin, 10000-mMargin)
            x += 1
            fx = CGFloat(x)*xDistance
            fy = CGFloat(y.y)*mVerticalRatioToDraw_view + mMargin
            //            drawEllipse(fx, fy, 2, 2, thickness: 2, UIColor.blue.cgColor)
            
            pointCloud.append(CGPoint.init(x: getX(fx+mMargin)!, y: getY(fy)!))
            
            drawEORect(fx, fy, 2, 2, thickness: 2, UIColor.blue.cgColor)
            
        }
    }
    
    fileprivate func roundRect(x: CGFloat, y: CGFloat, width: CGFloat, height: CGFloat)
    {
        let rectBgColor:     UIColor = UIColor.yellow
        let rectBorderColor: UIColor = UIColor.yellow
        let rectBorderWidth: CGFloat = 2
        let rectCornerRadius:CGFloat = 5
        
        let ctx: CGContext = UIGraphicsGetCurrentContext()!
        ctx.saveGState()
        
        ctx.setLineWidth(rectBorderWidth)
        ctx.setStrokeColor(rectBorderColor.cgColor)
        
        
        let rect = CGRect(x: x, y: y, width: width, height: height)
        let clipPath: CGPath = UIBezierPath(roundedRect: rect, cornerRadius: rectCornerRadius).cgPath
        let linePath: CGPath = UIBezierPath(roundedRect: rect, cornerRadius: rectCornerRadius).cgPath
        
        ctx.addPath(clipPath)
        ctx.setFillColor(rectBgColor.cgColor)
        ctx.closePath()
        ctx.fillPath()
        
        ctx.addPath(linePath)
        ctx.strokePath()
        
        ctx.restoreGState()
    }
    
    
    fileprivate func drawEORect(_ x: CGFloat, _ y: CGFloat, _ width: CGFloat, _ height: CGFloat, thickness: CGFloat, _ color: CGColor) {
        jhDraw.worldEllipse(context: mContext, getX(x+mMargin)!, getY(y)!, width, height, thickness, UIColor.red.cgColor)
        
        if GS.shared.current_eoGraphType == .general {
            for x in pointCloud {
                mContext?.setFillColor(jhDraw.jhColor(r: 184, g: 70, b: 201, a: 0.1))
                mContext?.setStrokeColor(jhDraw.jhColor(r: 184, g: 70, b: 201, a: 0.1))
                mContext?.setLineWidth(1)
                
                let rectangle = CGRect(x: x.x-5, y: x.y-20, width: 10, height: 40) //TODO: 좌표 계산 부분 한 곳으로 몰기.
                mContext?.addRect(rectangle)
                mContext?.drawPath(using: .fillStroke)
            }
        } else if GS.shared.current_eoGraphType == .first {
            for x in pointCloud {
                roundRect(x: x.x-5, y: x.y-20, width: 10, height: 40)
            }
        }
    }
}
